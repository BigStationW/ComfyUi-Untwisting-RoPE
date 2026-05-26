"""
UntwistingRoPE adapter: Flux / Flux2

Supports: Flux1-dev, Flux2-dev, Flux-Klein base, Flux-Klein distilled.

Patches DoubleStreamBlock.forward and SingleStreamBlock.forward directly,
mirroring the Z-Image _patch_joint_attention_modules approach so no changes
to __init__.py are needed.

Drop into ComfyUi-Untwisting-RoPE/models/ and restart ComfyUI.
"""
from __future__ import annotations

import traceback
import types
from typing import Any, List, Optional, Tuple

import torch

ARCHITECTURE = "flux"
DISPLAY_NAME = "Flux / Flux2"
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
_ACTIVE_ATTR = "_untwist_flux_active"


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

def matches_model(model_info: dict[str, Any]) -> bool:
    return str(model_info.get("image_model", "")).lower() in ("flux", "flux2")


def looks_like_diffusion_model(obj: Any) -> bool:
    return (
        obj is not None
        and hasattr(obj, "double_blocks")
        and hasattr(obj, "single_blocks")
        and hasattr(obj, "pe_embedder")
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
    raise RuntimeError("Could not find Flux/Flux2 diffusion model.")


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
        return dm.hidden_size // dm.num_heads
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
        from comfy.ldm.flux.math import apply_rope, apply_rope1
        from comfy.ldm.modules.attention import optimized_attention
    except ImportError as exc:
        raise RuntimeError(f"[Flux adapter] Could not import Flux math: {exc}")

    head_dim = _get_head_dim(dm)
    axes_dims = _get_axes_dims(dm)

    matched_double = matched_single = installed = restored = 0

    # ── DoubleStreamBlock ────────────────────────────────────────────────────
    for block_idx, block in enumerate(dm.double_blocks):
        if type(block).__name__ != "DoubleStreamBlock":
            continue
        matched_double += 1

        if hasattr(block, _ORIG_ATTR):
            block.forward = getattr(block, _ORIG_ATTR)
            restored += 1
        else:
            setattr(block, _ORIG_ATTR, block.forward)
        orig = getattr(block, _ORIG_ATTR)

        def make_double_forward(orig_fwd, blk_idx):
            def patched_forward(
                self, img, txt, vec, pe, attn_mask=None,
                modulation_dims_img=None, modulation_dims_txt=None,
                transformer_options={},
            ):
                cfg = (
                    transformer_options.get(config_key)
                    if isinstance(transformer_options, dict) else None
                )
                if not cfg or not cfg.get("enabled"):
                    return orig_fwd(
                        img, txt, vec, pe, attn_mask=attn_mask,
                        modulation_dims_img=modulation_dims_img,
                        modulation_dims_txt=modulation_dims_txt,
                        transformer_options=transformer_options,
                    )

                active_blocks = cfg.get("active_blocks", set())
                if active_blocks and blk_idx not in active_blocks:
                    return orig_fwd(
                        img, txt, vec, pe, attn_mask=attn_mask,
                        modulation_dims_img=modulation_dims_img,
                        modulation_dims_txt=modulation_dims_txt,
                        transformer_options=transformer_options,
                    )

                target_bsz = int(cfg.get("cross_batch_target_batch", 0))
                ref_ranges = cfg.get("ref_real_ranges") or cfg.get("ref_k_ranges") or []
                if target_bsz <= 0:
                    return orig_fwd(
                        img, txt, vec, pe, attn_mask=attn_mask,
                        modulation_dims_img=modulation_dims_img,
                        modulation_dims_txt=modulation_dims_txt,
                        transformer_options=transformer_options,
                    )

                bsz = img.shape[0]
                if bsz < target_bsz * 2:
                    return orig_fwd(
                        img, txt, vec, pe, attn_mask=attn_mask,
                        modulation_dims_img=modulation_dims_img,
                        modulation_dims_txt=modulation_dims_txt,
                        transformer_options=transformer_options,
                    )

                try:
                    return _double_block_cross_batch(
                        self, img, txt, vec, pe, attn_mask,
                        modulation_dims_img, modulation_dims_txt,
                        transformer_options, cfg, target_bsz, ref_ranges,
                        blk_idx, head_dim, axes_dims,
                        lerp, build_frequency_scale_vector, cross_batch_adain_qk,
                        apply_rope, apply_rope1, optimized_attention, config_key,
                    )
                except Exception as exc:
                    print(f"{prefix} ⚠ Flux double_block[{blk_idx}] patch failed: {exc}")
                    if cfg.get("verbose"):
                        traceback.print_exc()
                    return orig_fwd(
                        img, txt, vec, pe, attn_mask=attn_mask,
                        modulation_dims_img=modulation_dims_img,
                        modulation_dims_txt=modulation_dims_txt,
                        transformer_options=transformer_options,
                    )

            return patched_forward

        block.forward = types.MethodType(make_double_forward(orig, block_idx), block)
        setattr(block, _ACTIVE_ATTR, True)
        installed += 1

    # ── SingleStreamBlock ────────────────────────────────────────────────────
    for block_idx, block in enumerate(dm.single_blocks):
        if type(block).__name__ != "SingleStreamBlock":
            continue
        matched_single += 1

        if hasattr(block, _ORIG_ATTR):
            block.forward = getattr(block, _ORIG_ATTR)
            restored += 1
        else:
            setattr(block, _ORIG_ATTR, block.forward)
        orig = getattr(block, _ORIG_ATTR)

        def make_single_forward(orig_fwd, blk_idx):
            def patched_forward(
                self, x, vec, pe, attn_mask=None,
                modulation_dims=None, transformer_options={},
            ):
                cfg = (
                    transformer_options.get(config_key)
                    if isinstance(transformer_options, dict) else None
                )
                if not cfg or not cfg.get("enabled"):
                    return orig_fwd(x, vec, pe, attn_mask=attn_mask,
                                    modulation_dims=modulation_dims,
                                    transformer_options=transformer_options)

                active_blocks = cfg.get("active_blocks", set())
                if active_blocks and blk_idx not in active_blocks:
                    return orig_fwd(x, vec, pe, attn_mask=attn_mask,
                                    modulation_dims=modulation_dims,
                                    transformer_options=transformer_options)

                target_bsz = int(cfg.get("cross_batch_target_batch", 0))
                ref_ranges = cfg.get("ref_real_ranges") or cfg.get("ref_k_ranges") or []
                if target_bsz <= 0:
                    return orig_fwd(x, vec, pe, attn_mask=attn_mask,
                                    modulation_dims=modulation_dims,
                                    transformer_options=transformer_options)

                bsz = x.shape[0]
                if bsz < target_bsz * 2:
                    return orig_fwd(x, vec, pe, attn_mask=attn_mask,
                                    modulation_dims=modulation_dims,
                                    transformer_options=transformer_options)

                try:
                    return _single_block_cross_batch(
                        self, x, vec, pe, attn_mask, modulation_dims,
                        transformer_options, cfg, target_bsz, ref_ranges,
                        blk_idx, head_dim, axes_dims,
                        lerp, build_frequency_scale_vector, cross_batch_adain_qk,
                        apply_rope1, optimized_attention, config_key,
                    )
                except Exception as exc:
                    print(f"{prefix} ⚠ Flux single_block[{blk_idx}] patch failed: {exc}")
                    if cfg.get("verbose"):
                        traceback.print_exc()
                    return orig_fwd(x, vec, pe, attn_mask=attn_mask,
                                    modulation_dims=modulation_dims,
                                    transformer_options=transformer_options)

            return patched_forward

        block.forward = types.MethodType(make_single_forward(orig, block_idx), block)
        setattr(block, _ACTIVE_ATTR, True)
        installed += 1

    print(f"{prefix} Flux attention patch: "
          f"double={matched_double} single={matched_single} "
          f"installed={installed} restored={restored}")

    assert installed > 0, f"{prefix} FATAL: No Flux blocks patched."


# ─────────────────────────────────────────────────────────────────────────────
# Cross-batch attention logic
# ─────────────────────────────────────────────────────────────────────────────

def _double_block_cross_batch(
    self, img, txt, vec, pe, attn_mask,
    modulation_dims_img, modulation_dims_txt,
    transformer_options, cfg, target_bsz, ref_ranges,
    blk_idx, head_dim, axes_dims,
    lerp, build_frequency_scale_vector, cross_batch_adain_qk,
    apply_rope, apply_rope1, optimized_attention, config_key,
):
    from comfy.ldm.flux.layers import apply_mod

    # Modulation
    if self.modulation:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
    else:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

    # QKV projection + norm (same as original)
    img_modulated = apply_mod(self.img_norm1(img), (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
    img_qkv = self.img_attn.qkv(img_modulated)
    del img_modulated
    img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    del img_qkv
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    txt_modulated = apply_mod(self.txt_norm1(txt), (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    del txt_modulated
    txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    del txt_qkv
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    # Concatenate: [B, heads, txt+img, head_dim]
    q = torch.cat((txt_q, img_q), dim=2)
    k = torch.cat((txt_k, img_k), dim=2)
    v = torch.cat((txt_v, img_v), dim=2)
    del txt_q, img_q, txt_k, img_k, txt_v, img_v

    txt_len = txt.shape[1]
    seqlen = q.shape[2]

    progress = float(cfg.get("progress", 0.0))
    high_scale = lerp(cfg["high_scale_start"], cfg["high_scale_end"], progress)
    low_scale  = lerp(cfg["low_scale_start"],  cfg["low_scale_end"],  progress)
    beta       = float(cfg.get("beta", 2.0))
    hd         = int(cfg.get("head_dim", head_dim))
    axd        = cfg.get("axes_dims") or axes_dims

    # AdaIN on image Q/K (skip text tokens)
    if cfg.get("apply_adain") and float(cfg.get("adain_strength", 0)) > 0:
        q, k = _adain_img_tokens(q, k, cfg, target_bsz, float(cfg["adain_strength"]),
                                  txt_len, cross_batch_adain_qk)

    # RoPE
    q, k = apply_rope(q, k, pe)

    # Frequency scale
    scale_vec = build_frequency_scale_vector(hd, axd, high_scale, low_scale, beta,
                                              k.device, k.dtype).view(1, 1, 1, hd)

    # Reference K/V pieces (image tokens only, from ref_ranges)
    # Fall back to all image tokens if ref_ranges not populated by patchify hook
    img_len = seqlen - txt_len
    effective_ref_ranges = ref_ranges if ref_ranges else [(0, img_len)]
    ref_k_pieces, ref_v_pieces = [], []
    for s, e in effective_ref_ranges:
        # ref_ranges are image-token indices; offset by txt_len for joint sequence
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
        mask_t = _extend_mask(attn_mask, 0, target_bsz, ref_k_pieces)
    else:
        k_t, v_t, mask_t = k[:target_bsz], v[:target_bsz], _slice_mask(attn_mask, 0, target_bsz)

    out_t = _attn(q[:target_bsz], k_t, v_t, n_heads, mask_t, transformer_options)

    mask_r = _slice_mask(attn_mask, target_bsz, target_bsz * 2)
    out_r = _attn(q[target_bsz:target_bsz * 2],
                  k[target_bsz:target_bsz * 2],
                  v[target_bsz:target_bsz * 2],
                  n_heads, mask_r, transformer_options)

    outs = [out_t, out_r]
    if q.shape[0] > target_bsz * 2:
        outs.append(_attn(q[target_bsz * 2:], k[target_bsz * 2:], v[target_bsz * 2:],
                          n_heads, None, transformer_options))

    attn = torch.cat(outs, dim=0)
    del q, k, v

    txt_attn = attn[:, :txt_len]
    img_attn = attn[:, txt_len:]

    img += apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
    del img_attn
    img += apply_mod(
        self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)),
        img_mod2.gate, None, modulation_dims_img,
    )

    txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
    del txt_attn
    txt += apply_mod(
        self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)),
        txt_mod2.gate, None, modulation_dims_txt,
    )

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)
    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)
    return img, txt


def _single_block_cross_batch(
    self, x, vec, pe, attn_mask, modulation_dims,
    transformer_options, cfg, target_bsz, ref_ranges,
    blk_idx, head_dim, axes_dims,
    lerp, build_frequency_scale_vector, cross_batch_adain_qk,
    apply_rope1, optimized_attention, config_key,
):
    from comfy.ldm.flux.layers import apply_mod

    if self.modulation:
        mod, _ = self.modulation(vec)
    else:
        mod = vec

    qkv, mlp = torch.split(
        self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)),
        [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1,
    )

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    del qkv
    q, k = self.norm(q, k, v)

    seqlen = q.shape[2]

    progress = float(cfg.get("progress", 0.0))
    high_scale = lerp(cfg["high_scale_start"], cfg["high_scale_end"], progress)
    low_scale  = lerp(cfg["low_scale_start"],  cfg["low_scale_end"],  progress)
    beta       = float(cfg.get("beta", 2.0))
    hd         = int(cfg.get("head_dim", head_dim))
    axd        = cfg.get("axes_dims") or axes_dims

    if cfg.get("apply_adain") and float(cfg.get("adain_strength", 0)) > 0:
        # SingleStream: no txt/img split; apply to full sequence
        q_t = q.movedim(1, 2)
        k_t = k.movedim(1, 2)
        q_t, k_t = cross_batch_adain_qk(q_t, k_t, cfg, target_bsz, float(cfg["adain_strength"]))
        q = q_t.movedim(2, 1)
        k = k_t.movedim(2, 1)

    q = apply_rope1(q, pe)
    k = apply_rope1(k, pe)

    scale_vec = build_frequency_scale_vector(hd, axd, high_scale, low_scale, beta,
                                              k.device, k.dtype).view(1, 1, 1, hd)

    effective_ref_ranges = ref_ranges if ref_ranges else [(0, seqlen)]
    ref_k_pieces, ref_v_pieces = [], []
    for s, e in effective_ref_ranges:
        s, e = max(0, int(s)), min(int(e), seqlen)
        if e <= s:
            continue
        ref_k_pieces.append(k[target_bsz:target_bsz * 2, :, s:e, :] * scale_vec)
        ref_v_pieces.append(v[target_bsz:target_bsz * 2, :, s:e, :])

    n_heads = q.shape[1]

    if ref_k_pieces:
        k_t = torch.cat([k[:target_bsz]] + ref_k_pieces, dim=2)
        v_t = torch.cat([v[:target_bsz]] + ref_v_pieces, dim=2)
        mask_t = _extend_mask(attn_mask, 0, target_bsz, ref_k_pieces)
    else:
        k_t, v_t, mask_t = k[:target_bsz], v[:target_bsz], _slice_mask(attn_mask, 0, target_bsz)

    out_t = _attn(q[:target_bsz], k_t, v_t, n_heads, mask_t, transformer_options)
    out_r = _attn(
        q[target_bsz:target_bsz * 2],
        k[target_bsz:target_bsz * 2],
        v[target_bsz:target_bsz * 2],
        n_heads, _slice_mask(attn_mask, target_bsz, target_bsz * 2), transformer_options,
    )

    outs = [out_t, out_r]
    if q.shape[0] > target_bsz * 2:
        outs.append(_attn(q[target_bsz * 2:], k[target_bsz * 2:], v[target_bsz * 2:],
                          n_heads, None, transformer_options))

    attn = torch.cat(outs, dim=0)
    del q, k, v

    if self.yak_mlp:
        mlp = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2:]) * mlp[..., :self.mlp_hidden_dim_first // 2]
    else:
        mlp = self.mlp_act(mlp)

    output = self.linear2(torch.cat((attn, mlp), 2))
    x += apply_mod(output, mod.gate, None, modulation_dims)
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Shared attention and mask helpers
# ─────────────────────────────────────────────────────────────────────────────

def _attn(q, k, v, n_heads, mask, transformer_options):
    from comfy.ldm.modules.attention import optimized_attention
    # skip_reshape=True expects [B, heads, seq, head_dim] — no transpose needed
    return optimized_attention(
        q, k, v,
        n_heads, skip_reshape=True, mask=mask,
        transformer_options=transformer_options,
    )


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


def _adain_img_tokens(q, k, cfg, target_bsz, strength, txt_len, cross_batch_adain_qk):
    """Apply AdaIN to image tokens only. q/k: [B, heads, seq, head_dim]."""
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