"""
UntwistingRoPE adapter: Qwen image generation model.

Patches QwenImageTransformerBlock's internal Attention.forward directly,
since Qwen applies apply_rope1 *after* returning from attn1_patch (so
attn1_patch alone can't control RoPE-then-scale ordering).

Drop into ComfyUi-Untwisting-RoPE/models/ and restart ComfyUI.
"""
from __future__ import annotations

import traceback
import types
from typing import Any, List, Optional, Tuple

import torch

ARCHITECTURE = "qwen_image"
DISPLAY_NAME = "Qwen Image"
PRIORITY = 10

CONFIG_KEY = "untwisting_rope"

DIFFUSION_ATTR_PATHS = (
    "diffusion_model",
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
)
SEARCH_CHILD_ATTRS = ("model", "inner_model", "diffusion_model", "unet", "wrapped")

_ORIG_ATTR = "_untwist_orig_forward"
_ACTIVE_ATTR = "_untwist_qwen_active"


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

def matches_model(model_info: dict[str, Any]) -> bool:
    return str(model_info.get("image_model", "")).lower() == "qwen_image"


def looks_like_diffusion_model(obj: Any) -> bool:
    return (
        obj is not None
        and hasattr(obj, "transformer_blocks")
        and hasattr(obj, "pe_embedder")
        and not hasattr(obj, "double_blocks")   # exclude Flux
    )


def _roots(mp: Any) -> list[Any]:
    roots: list[Any] = []
    if hasattr(mp, "model"):
        roots.append(mp.model)
    roots.append(mp)
    return roots


def _get_attr_path(root: Any, path: str) -> Tuple[Any, bool]:
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None, False
        obj = getattr(obj, part)
    return obj, True


def find_diffusion_model(model_patcher: Any) -> Any:
    roots = _roots(model_patcher)
    for root in roots:
        for path in DIFFUSION_ATTR_PATHS:
            obj, ok = _get_attr_path(root, path)
            if ok and looks_like_diffusion_model(obj):
                return obj
    seen: set[int] = set()
    stack = list(roots)
    while stack and len(seen) < 256:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if looks_like_diffusion_model(obj):
            return obj
        for name in SEARCH_CHILD_ATTRS:
            if hasattr(obj, name):
                try:
                    stack.append(getattr(obj, name))
                except Exception:
                    pass
    raise RuntimeError("Could not find Qwen image diffusion model.")


# ─────────────────────────────────────────────────────────────────────────────
# Architecture helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_axes_dims(dm: Any) -> List[int]:
    try:
        return list(dm.pe_embedder.axes_dim)
    except Exception:
        return [16, 56, 56]


def _get_head_dim(dm: Any) -> int:
    try:
        block = dm.transformer_blocks[0]
        attn = block.attn if hasattr(block, "attn") else block.attention
        return attn.to_q.out_features // attn.heads
    except Exception:
        return 128


def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {"architecture": ARCHITECTURE}
    if dm is not None:
        cfg["axes_dims"] = _get_axes_dims(dm)
        cfg["head_dim"] = _get_head_dim(dm)
    return cfg


def uses_reference_branch_kv() -> bool:
    return True


def prepare_reference_conditioning(
    ref_conditioning: Any,
    dm: Any,
    device: Any,
    dtype: Any,
    stats: Any = None,
    label: str = "",
    helpers: dict[str, Any] | None = None,
) -> Tuple[Any, str]:
    return ref_conditioning, "not-applicable"


# ─────────────────────────────────────────────────────────────────────────────
# Module patching
# ─────────────────────────────────────────────────────────────────────────────

def patch_attention_modules(dm: Any, stats: Any, helpers: dict[str, Any] | None = None) -> None:
    helpers = helpers or {}
    prefix = helpers.get("prefix", "[UntwistingRoPE]")
    config_key = helpers.get("config_key", CONFIG_KEY)
    lerp = helpers["lerp"]
    build_frequency_scale_vector = helpers["build_frequency_scale_vector"]
    cross_batch_adain_qk = helpers["cross_batch_adain_qk"]

    try:
        from comfy.ldm.flux.math import apply_rope1
        from comfy.ldm.modules.attention import optimized_attention_masked
    except ImportError as exc:
        raise RuntimeError(f"[Qwen adapter] Could not import math utils: {exc}")

    head_dim = _get_head_dim(dm)
    axes_dims = _get_axes_dims(dm)

    matched = installed = restored = 0

    for block_idx, block in enumerate(dm.transformer_blocks):
        # Find the Attention sub-module (named 'attn' or 'attention')
        attn_mod = getattr(block, "attn", None) or getattr(block, "attention", None)
        if attn_mod is None:
            continue
        if type(attn_mod).__name__ != "Attention":
            continue

        matched += 1

        if hasattr(attn_mod, _ORIG_ATTR):
            attn_mod.forward = getattr(attn_mod, _ORIG_ATTR)
            restored += 1
        else:
            setattr(attn_mod, _ORIG_ATTR, attn_mod.forward)
        orig = getattr(attn_mod, _ORIG_ATTR)

        def make_forward(orig_fwd, blk_idx, hd, axd):
            def patched_forward(
                self,
                hidden_states,
                encoder_hidden_states=None,
                encoder_hidden_states_mask=None,
                attention_mask=None,
                image_rotary_emb=None,
                transformer_options={},
            ):
                cfg = (
                    transformer_options.get(config_key)
                    if isinstance(transformer_options, dict) else None
                )
                if not cfg or not cfg.get("enabled"):
                    return orig_fwd(
                        hidden_states, encoder_hidden_states,
                        encoder_hidden_states_mask, attention_mask,
                        image_rotary_emb, transformer_options,
                    )

                active_blocks = cfg.get("active_blocks", set())
                if active_blocks and blk_idx not in active_blocks:
                    return orig_fwd(
                        hidden_states, encoder_hidden_states,
                        encoder_hidden_states_mask, attention_mask,
                        image_rotary_emb, transformer_options,
                    )

                target_bsz = int(cfg.get("cross_batch_target_batch", 0))
                ref_ranges = cfg.get("ref_real_ranges") or cfg.get("ref_k_ranges") or []
                if target_bsz <= 0:
                    return orig_fwd(
                        hidden_states, encoder_hidden_states,
                        encoder_hidden_states_mask, attention_mask,
                        image_rotary_emb, transformer_options,
                    )

                bsz = hidden_states.shape[0]
                if bsz < target_bsz * 2:
                    return orig_fwd(
                        hidden_states, encoder_hidden_states,
                        encoder_hidden_states_mask, attention_mask,
                        image_rotary_emb, transformer_options,
                    )

                try:
                    return _qwen_cross_batch_attention(
                        self, hidden_states, encoder_hidden_states,
                        encoder_hidden_states_mask, attention_mask,
                        image_rotary_emb, transformer_options,
                        cfg, target_bsz, ref_ranges, blk_idx, hd, axd,
                        lerp, build_frequency_scale_vector, cross_batch_adain_qk,
                        apply_rope1, optimized_attention_masked,
                    )
                except Exception as exc:
                    print(f"{prefix} ⚠ Qwen block[{blk_idx}] patch failed: {exc}")
                    if cfg.get("verbose"):
                        traceback.print_exc()
                    return orig_fwd(
                        hidden_states, encoder_hidden_states,
                        encoder_hidden_states_mask, attention_mask,
                        image_rotary_emb, transformer_options,
                    )

            return patched_forward

        attn_mod.forward = types.MethodType(
            make_forward(orig, block_idx, head_dim, axes_dims), attn_mod
        )
        setattr(attn_mod, _ACTIVE_ATTR, True)
        installed += 1

    print(f"{prefix} Qwen attention patch: "
          f"matched={matched} installed={installed} restored={restored}")
    assert installed > 0, f"{prefix} FATAL: No Qwen Attention modules patched."


# ─────────────────────────────────────────────────────────────────────────────
# Cross-batch attention
# ─────────────────────────────────────────────────────────────────────────────

def _qwen_cross_batch_attention(
    self,
    hidden_states,          # image tokens [B, seq_img, dim]
    encoder_hidden_states,  # text tokens  [B, seq_txt, dim]
    encoder_hidden_states_mask,
    attention_mask,
    image_rotary_emb,       # pe: [B, 1, seq_txt+seq_img, head_dim//2, 2, 2]
    transformer_options,
    cfg, target_bsz, ref_ranges, blk_idx, head_dim, axes_dims,
    lerp, build_frequency_scale_vector, cross_batch_adain_qk,
    apply_rope1, optimized_attention_masked,
):
    batch_size = hidden_states.shape[0]
    seq_img = hidden_states.shape[1]
    seq_txt = encoder_hidden_states.shape[1]

    # Project and reshape to [B, heads, seq, head_dim]
    img_q = self.to_q(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
    img_k = self.to_k(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
    img_v = self.to_v(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2)

    txt_q = self.add_q_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
    txt_k = self.add_k_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
    txt_v = self.add_v_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2)

    img_q = self.norm_q(img_q)
    img_k = self.norm_k(img_k)
    txt_q = self.norm_added_q(txt_q)
    txt_k = self.norm_added_k(txt_k)

    # Joint sequence [txt | img]
    q = torch.cat([txt_q, img_q], dim=2)
    k = torch.cat([txt_k, img_k], dim=2)
    v = torch.cat([txt_v, img_v], dim=2)
    del txt_q, img_q, txt_k, img_k, txt_v, img_v

    txt_len = seq_txt
    seqlen = q.shape[2]

    progress = float(cfg.get("progress", 0.0))
    high_scale = lerp(cfg["high_scale_start"], cfg["high_scale_end"], progress)
    low_scale  = lerp(cfg["low_scale_start"],  cfg["low_scale_end"],  progress)
    beta       = float(cfg.get("beta", 2.0))
    hd         = int(cfg.get("head_dim", head_dim))
    axd        = cfg.get("axes_dims") or axes_dims

    # Optional AdaIN on image Q/K (skip text prefix)
    if cfg.get("apply_adain") and float(cfg.get("adain_strength", 0)) > 0:
        q, k = _adain_img_tokens(q, k, cfg, target_bsz, float(cfg["adain_strength"]),
                                  txt_len, cross_batch_adain_qk)

    # RoPE
    q = apply_rope1(q, image_rotary_emb)
    k = apply_rope1(k, image_rotary_emb)

    # Frequency scale vector
    scale_vec = build_frequency_scale_vector(
        hd, axd, high_scale, low_scale, beta, k.device, k.dtype,
    ).view(1, 1, 1, hd)

    # Build attention mask
    if encoder_hidden_states_mask is not None:
        attn_mask_full = torch.zeros(
            (batch_size, 1, seq_txt + seq_img),
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        attn_mask_full[:, 0, :seq_txt] = encoder_hidden_states_mask
    else:
        attn_mask_full = None

    # Reference K/V pieces (image tokens offset by txt_len in joint sequence)
    # Fall back to full image-token range if ref_ranges not populated by patchify hook
    effective_ref_ranges = ref_ranges if ref_ranges else [(0, seq_img)]
    ref_k_pieces, ref_v_pieces = [], []
    for s, e in effective_ref_ranges:
        s_j = max(0, int(s) + txt_len)
        e_j = min(int(e) + txt_len, seqlen)
        if e_j <= s_j:
            continue
        ref_k_pieces.append(k[target_bsz:target_bsz * 2, :, s_j:e_j, :] * scale_vec)
        ref_v_pieces.append(v[target_bsz:target_bsz * 2, :, s_j:e_j, :])

    n_heads = q.shape[1]

    if ref_k_pieces:
        k_t = torch.cat([k[:target_bsz]] + ref_k_pieces, dim=2)
        v_t = torch.cat([v[:target_bsz]] + ref_v_pieces, dim=2)
        mask_t = _extend_mask(attn_mask_full, 0, target_bsz, ref_k_pieces)
    else:
        k_t, v_t = k[:target_bsz], v[:target_bsz]
        mask_t = _slice_mask(attn_mask_full, 0, target_bsz)

    out_t = optimized_attention_masked(
        q[:target_bsz],
        k_t,
        v_t,
        n_heads, mask_t, skip_reshape=True,
        transformer_options=transformer_options,
    )

    out_r = optimized_attention_masked(
        q[target_bsz:target_bsz * 2],
        k[target_bsz:target_bsz * 2],
        v[target_bsz:target_bsz * 2],
        n_heads, _slice_mask(attn_mask_full, target_bsz, target_bsz * 2),
        skip_reshape=True, transformer_options=transformer_options,
    )

    outs = [out_t, out_r]
    if q.shape[0] > target_bsz * 2:
        outs.append(optimized_attention_masked(
            q[target_bsz * 2:],
            k[target_bsz * 2:],
            v[target_bsz * 2:],
            n_heads, None, skip_reshape=True,
            transformer_options=transformer_options,
        ))

    joint_out = torch.cat(outs, dim=0)  # [B, seq_txt+seq_img, dim]
    del q, k, v

    txt_out = joint_out[:, :seq_txt, :]
    img_out = joint_out[:, seq_txt:, :]

    img_out = self.to_out[0](img_out)
    img_out = self.to_out[1](img_out)
    txt_out = self.to_add_out(txt_out)

    return img_out, txt_out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _adain_img_tokens(q, k, cfg, target_bsz, strength, txt_len, cross_batch_adain_qk):
    q = q.clone()
    k = k.clone()
    seqlen = q.shape[2]
    s, e = txt_len, seqlen
    if e <= s:
        return q, k
    q_img = q[:, :, s:e, :].movedim(1, 2)
    k_img = k[:, :, s:e, :].movedim(1, 2)
    q_img, k_img = cross_batch_adain_qk(q_img, k_img, cfg, target_bsz, strength)
    q[:, :, s:e, :] = q_img.movedim(2, 1)
    k[:, :, s:e, :] = k_img.movedim(2, 1)
    return q, k


def _extend_mask(attn_mask, start, end, ref_k_pieces):
    if attn_mask is None:
        return None
    try:
        ref_len = sum(int(p.shape[2]) for p in ref_k_pieces)
        mt = attn_mask[start:end]
        if mt.ndim >= 2:
            pad = torch.zeros((*mt.shape[:-1], ref_len), device=mt.device, dtype=mt.dtype)
            return torch.cat([mt, pad], dim=-1)
    except Exception:
        pass
    return None


def _slice_mask(attn_mask, start, end):
    if attn_mask is None:
        return None
    try:
        if int(attn_mask.shape[0]) >= end:
            return attn_mask[start:end]
    except Exception:
        pass
    return None