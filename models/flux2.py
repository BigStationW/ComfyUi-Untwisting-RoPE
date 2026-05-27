from __future__ import annotations

import traceback
import types
from typing import Any, Iterable, List, Optional, Tuple

import torch
from comfy.ldm.flux.math import apply_rope, attention as flux_attention
from comfy.ldm.modules.attention import optimized_attention_masked


# ---------------------------------------------------------------------------
# Required adapter identity
# ---------------------------------------------------------------------------

ARCHITECTURE = "flux2"
DISPLAY_NAME = "FLUX.2"

# Strict ComfyUI supported_models class names only.
SUPPORTED_MODEL_CONFIG_CLASSES: set[str] = {"Flux2"}

_DIFFUSION_ATTR_PATHS = (
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
    "diffusion_model",
)


# ---------------------------------------------------------------------------
# Safe metadata / lookup helpers
# ---------------------------------------------------------------------------

def _get_attr_path(root: Any, attr_path: str) -> tuple[Any, bool]:
    obj = root
    for part in attr_path.split("."):
        if obj is None or not hasattr(obj, part):
            return None, False
        try:
            obj = getattr(obj, part)
        except Exception:
            return None, False
    return obj, True


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default


def _sequence_of_ints(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for item in value:
        i = _safe_int(item)
        if i is None:
            return []
        out.append(i)
    return out


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "y", "t")
    return bool(value)


def _coerce_strength01(value: Any, default: float = 0.0) -> float:
    try:
        strength = float(value)
    except Exception:
        strength = float(default)
    if not torch.isfinite(torch.tensor(strength)):
        strength = float(default)
    return max(0.0, min(1.0, strength))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


# ---------------------------------------------------------------------------
# Required hooks
# ---------------------------------------------------------------------------

def matches_model(model_info: dict[str, Any]) -> bool:
    """Select FLUX.2 only from ComfyUI's explicit MODEL metadata."""
    return str(model_info.get("model_config_class", "")) in SUPPORTED_MODEL_CONFIG_CLASSES


def is_model_identity(model_info: dict[str, Any]) -> bool:
    """Backward-compatible alias for older callers."""
    return matches_model(model_info)


def find_diffusion_model(model_patcher: Any) -> Any:
    """Return ComfyUI BaseModel.diffusion_model after metadata selected this adapter."""
    for path in _DIFFUSION_ATTR_PATHS:
        obj, ok = _get_attr_path(model_patcher, path)
        if ok and obj is not None:
            return obj
    raise RuntimeError(f"Could not find ComfyUI BaseModel.diffusion_model for {DISPLAY_NAME}.")


# ---------------------------------------------------------------------------
# Runtime config helpers
# ---------------------------------------------------------------------------

def axes_dims_from_dm(dm: Any | None) -> list[int]:
    """Read FLUX.2 RoPE axes from the already-selected ComfyUI diffusion module."""
    if dm is None:
        return []
    params = getattr(dm, "params", None)
    axes = _sequence_of_ints(getattr(params, "axes_dim", None))
    if axes:
        return axes
    return []


def head_dim_from_dm(dm: Any | None) -> int | None:
    """Read FLUX.2 head_dim from explicit diffusion-module params after selection."""
    if dm is None:
        return None
    params = getattr(dm, "params", None)
    hidden = _safe_int(getattr(params, "hidden_size", None))
    heads = _safe_int(getattr(params, "num_heads", None))
    if hidden is None or heads is None or heads <= 0:
        return None
    if hidden % heads != 0:
        return None
    return hidden // heads


def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    """Architecture-specific config merged into transformer_options."""
    cfg: dict[str, Any] = {"architecture": ARCHITECTURE}

    axes = axes_dims_from_dm(dm)
    if axes:
        cfg["axes_dims"] = axes

    head_dim = head_dim_from_dm(dm)
    if head_dim is not None:
        cfg["head_dim"] = head_dim

    return cfg


# ---------------------------------------------------------------------------
# FLUX.2 patch-target helpers
# ---------------------------------------------------------------------------

def _index_in_range(parts: list[str], min_layer: int, max_layer: int) -> bool:
    if len(parts) < 2:
        return False
    idx = _safe_int(parts[1])
    if idx is None:
        return False
    return int(min_layer) <= idx <= int(max_layer)


def is_double_stream_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    """FLUX.2 double-stream attention modules are double_blocks.N.{img_attn,txt_attn}."""
    parts = str(name).split(".")
    if len(parts) != 3:
        return False
    if parts[0] != "double_blocks" or parts[2] not in {"img_attn", "txt_attn"}:
        return False
    return _index_in_range(parts, min_layer, max_layer)


def is_single_stream_block_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    """FLUX.2 single-stream injection must target the whole single_blocks.N block."""
    parts = str(name).split(".")
    if len(parts) != 2:
        return False
    if parts[0] != "single_blocks":
        return False
    return _index_in_range(parts, min_layer, max_layer)


def is_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    """Names a verified FLUX.2 target after metadata selected this adapter."""
    return (
        is_double_stream_attention_name(name, min_layer, max_layer)
        or is_single_stream_block_name(name, min_layer, max_layer)
    )


def block_index_from_name(name: str) -> int:
    parts = str(name).split(".")
    if parts and parts[0] in {"double_blocks", "single_blocks"} and len(parts) >= 2:
        idx = _safe_int(parts[1], -1)
        return -1 if idx is None else idx
    return -1


def stream_kind_from_name(name: str) -> str:
    if is_single_stream_block_name(name):
        return "single"
    if is_double_stream_attention_name(name):
        suffix = str(name).split(".")[-1]
        return "double_img" if suffix == "img_attn" else "double_txt"
    return "unknown"


def is_flux2_self_attention_module(module: Any) -> bool:
    """Predicate for ComfyUI FLUX SelfAttention modules after metadata selection."""
    required_attrs = ("qkv", "norm", "proj", "num_heads")
    return all(hasattr(module, attr) for attr in required_attrs) and callable(getattr(module, "forward", None))


def is_flux2_single_stream_block(module: Any) -> bool:
    """Predicate for ComfyUI FLUX SingleStreamBlock modules after metadata selection."""
    required_attrs = ("linear1", "linear2", "norm", "hidden_size", "num_heads")
    return all(hasattr(module, attr) for attr in required_attrs) and callable(getattr(module, "forward", None))


def is_joint_attention(module: Any) -> bool:
    """Compatibility predicate for adapter-aware callers."""
    return is_flux2_self_attention_module(module) or is_flux2_single_stream_block(module)


def iter_flux2_patch_targets(dm: Any, min_layer: int = 0, max_layer: int = 999):
    """Yield (name, module, stream_kind, block_index) for verified FLUX.2 targets."""
    for name, module in dm.named_modules():
        if not is_attention_name(name, min_layer, max_layer):
            continue
        if is_single_stream_block_name(name, min_layer, max_layer):
            if not is_flux2_single_stream_block(module):
                continue
        elif not is_flux2_self_attention_module(module):
            continue
        yield name, module, stream_kind_from_name(name), block_index_from_name(name)


# ---------------------------------------------------------------------------
# Local reusable math/AdaIN helpers for the adapter-owned patch
# ---------------------------------------------------------------------------

def _adain(target: torch.Tensor, style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    t_mean = target.mean(dim=1, keepdim=True)
    s_mean = style.mean(dim=1, keepdim=True)
    t_std = target.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    s_std = style.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    return (target - t_mean) / t_std * s_std + s_mean


def _local_cross_batch_adain_qk(xq, xk, cfg, target_bsz, strength, eps=1e-6, xv=None):
    """Fallback copy of the model-neutral helper; tensors are [B,S,H,D]."""
    return_v = xv is not None
    if target_bsz <= 0 or xq.shape[0] < target_bsz * 2:
        return (xq, xk, xv) if return_v else (xq, xk)
    a = max(0.0, min(1.0, float(strength)))
    if a <= 0.0:
        return (xq, xk, xv) if return_v else (xq, xk)
    seqlen = int(xq.shape[1])
    apply_v = return_v and _coerce_bool(cfg.get("adain_on_v", False))
    for s, e in (cfg.get("target_qk_adain_ranges") or []):
        s, e = max(0, int(s)), min(int(e), seqlen)
        if e <= s:
            continue
        q_t, k_t = xq[:target_bsz, s:e], xk[:target_bsz, s:e]
        q_r, k_r = xq[target_bsz:target_bsz * 2, s:e], xk[target_bsz:target_bsz * 2, s:e]
        xq[:target_bsz, s:e] = q_t * (1 - a) + _adain(q_t, q_r, eps) * a
        xk[:target_bsz, s:e] = k_t * (1 - a) + _adain(k_t, k_r, eps) * a
        if apply_v:
            v_t = xv[:target_bsz, s:e]
            v_r = xv[target_bsz:target_bsz * 2, s:e]
            xv[:target_bsz, s:e] = v_t * (1 - a) + _adain(v_t, v_r, eps) * a
    return (xq, xk, xv) if return_v else (xq, xk)


def _local_build_frequency_scale_vector(head_dim, axes_dims, high_scale, low_scale, beta, device, dtype):
    if not axes_dims or sum(int(x) for x in axes_dims) != int(head_dim):
        axes_dims = [int(head_dim)]
    axes_dims = [int(x) for x in axes_dims]
    pieces: list[torch.Tensor] = []
    for axis_dim in axes_dims:
        n_pairs = axis_dim // 2
        if n_pairs <= 0:
            pieces.append(torch.ones(axis_dim, device=device, dtype=dtype))
            continue
        d_tilde = (
            torch.zeros(1, device=device, dtype=torch.float32)
            if n_pairs == 1 else
            torch.linspace(0.0, 1.0, n_pairs, device=device, dtype=torch.float32)
        )
        pair_scales = float(high_scale) + (float(low_scale) - float(high_scale)) * d_tilde.pow(float(beta))
        pieces.append(pair_scales.to(dtype=dtype).repeat_interleave(2))
        if axis_dim % 2:
            pieces.append(torch.ones(1, device=device, dtype=dtype))
    out = torch.cat(pieces, dim=0)
    if out.numel() >= int(head_dim):
        return out[: int(head_dim)]
    return torch.nn.functional.pad(out, (0, int(head_dim) - out.numel()), value=1.0)


def _flux_kv_heads_if_needed(k: torch.Tensor, v: torch.Tensor, q_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand KV heads for FLUX tensors in [B,H,S,D] layout."""
    kv = int(k.shape[1])
    qh = int(q_heads)
    if kv == qh:
        return k, v
    if qh % kv != 0:
        raise RuntimeError(f"Cannot expand FLUX KV heads: q={qh}, kv={kv}")
    n = qh // kv
    k = k.unsqueeze(2).repeat(1, 1, n, 1, 1).flatten(1, 2)
    v = v.unsqueeze(2).repeat(1, 1, n, 1, 1).flatten(1, 2)
    return k, v


def _flux_append_ref_padding_to_mask(mask: Any, target_bsz: int, ref_len: int):
    """Append valid additive-mask slots for injected reference K/V tokens."""
    if mask is None or ref_len <= 0:
        return None
    try:
        mask_t = mask[:target_bsz]
        if mask_t.ndim >= 2:
            padding = torch.zeros(
                (*mask_t.shape[:-1], int(ref_len)),
                device=mask_t.device,
                dtype=mask_t.dtype,
            )
            return torch.cat([mask_t, padding], dim=-1)
    except Exception:
        return None
    return None


def _flux_slice_mask(mask: Any, start: int, end: int):
    if mask is None:
        return None
    try:
        return mask[int(start): int(end)]
    except Exception:
        return None


def _flux_adain_qkv_for_image_range(q, k, v, cfg, target_bsz: int, image_range, cross_batch_adain_qk):
    """Run AdaIN on FLUX [B,H,S,D] tensors over image tokens."""
    if not cfg.get("apply_adain") or float(cfg.get("adain_strength", 0.0)) <= 0.0:
        return q, k, v

    s, e = int(image_range[0]), int(image_range[1])
    if e <= s:
        return q, k, v

    # Existing helper expects [B,S,H,D]. FLUX attention uses [B,H,S,D].
    q_sh = q.movedim(1, 2).clone()
    k_sh = k.movedim(1, 2).clone()
    v_sh = v.movedim(1, 2).clone() if _coerce_bool(cfg.get("adain_on_v", False)) else None

    cfg_local = dict(cfg)
    cfg_local["target_qk_adain_ranges"] = [(s, e)]
    out = cross_batch_adain_qk(
        q_sh, k_sh, cfg_local, target_bsz, float(cfg["adain_strength"]), xv=v_sh
    )

    if v_sh is not None:
        q_sh, k_sh, v_sh = out
        v = v_sh.movedim(1, 2)
    else:
        q_sh, k_sh = out
    return q_sh.movedim(1, 2), k_sh.movedim(1, 2), v


# ---------------------------------------------------------------------------
# Optional conditioning / patch hooks
# ---------------------------------------------------------------------------

def prepare_reference_conditioning(
    ref_conditioning: Any,
    dm: Any,
    device: Any,
    dtype: Any,
    stats: Any = None,
    label: str = "",
    helpers: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """FLUX.2 reference conditioning does not need adapter preprocessing here."""
    return ref_conditioning, "not-applicable"


def patch_attention_modules(
    dm: Any,
    stats: Any,
    helpers: dict[str, Any] | None = None,
) -> tuple[int, int, int, list[str]]:
    """Patch ComfyUI FLUX/FLUX.2 DoubleStreamBlock and SingleStreamBlock forwards.

    This architecture-specific implementation intentionally lives in the adapter
    file. The top-level node only supplies generic helpers through ``helpers``.
    """
    helpers = helpers or {}
    prefix = str(helpers.get("prefix", "[UntwistingRoPE]"))
    config_key = str(helpers.get("config_key", "untwisting_rope"))
    lerp = helpers.get("lerp") if callable(helpers.get("lerp")) else _lerp
    cross_batch_adain_qk = (
        helpers.get("cross_batch_adain_qk")
        if callable(helpers.get("cross_batch_adain_qk")) else
        _local_cross_batch_adain_qk
    )
    build_frequency_scale_vector = (
        helpers.get("build_frequency_scale_vector")
        if callable(helpers.get("build_frequency_scale_vector")) else
        _local_build_frequency_scale_vector
    )

    try:
        from comfy.ldm.flux.layers import apply_mod
    except Exception as exc:
        raise RuntimeError(f"{prefix} Could not import FLUX layers.apply_mod: {exc}")

    def _verbose_enabled() -> bool:
        return _coerce_bool(getattr(stats, "verbose", False)) or _coerce_bool(getattr(stats, "rf_verbose", False))

    def _reference_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pe: torch.Tensor,
        attn_mask: Any,
        transformer_options: dict[str, Any],
        image_range: tuple[int, int],
        module_name: str,
    ) -> torch.Tensor:
        """FLUX attention replacement for ComfyUI tensors in [B,H,S,D]."""
        try:
            cfg = (
                transformer_options.get(config_key)
                if isinstance(transformer_options, dict) else None
            )
            if not cfg or not cfg.get("enabled"):
                return flux_attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)

            target_bsz = int(cfg.get("cross_batch_target_batch", 0))
            if target_bsz <= 0 or q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
                return flux_attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)

            bsz, q_heads, seqlen, head_dim = q.shape
            if bsz < target_bsz * 2:
                return flux_attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)

            block_idx = int(transformer_options.get("block_index", -1))
            active_blocks = cfg.get("active_blocks", set())
            if active_blocks and block_idx not in active_blocks:
                return flux_attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)

            img_s, img_e = int(image_range[0]), int(image_range[1])
            img_s = max(0, min(img_s, int(seqlen)))
            img_e = max(img_s, min(img_e, int(seqlen)))
            if img_e <= img_s:
                return flux_attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)

            if hasattr(stats, "attn_calls"):
                stats.attn_calls += 1
            if hasattr(stats, "adapter_attn_calls"):
                stats.adapter_attn_calls += 1

            q, k, v = _flux_adain_qkv_for_image_range(
                q, k, v, cfg, target_bsz, (img_s, img_e), cross_batch_adain_qk
            )

            # Apply RoPE before appending reference K/V, matching ComfyUI's FLUX attention path.
            q, k = apply_rope(q, k, pe)

            progress = float(cfg.get("progress", 0.0))
            high_scale = lerp(cfg["high_scale_start"], cfg["high_scale_end"], progress)
            low_scale = lerp(cfg["low_scale_start"], cfg["low_scale_end"], progress)
            beta = float(cfg.get("beta", 2.0))
            scale_vec = build_frequency_scale_vector(
                int(head_dim),
                cfg.get("axes_dims") or axes_dims_from_dm(dm),
                high_scale,
                low_scale,
                beta,
                k.device,
                k.dtype,
            ).view(1, 1, 1, int(head_dim))

            ref_k = k[target_bsz:target_bsz * 2, :, img_s:img_e, :] * scale_vec
            ref_v = v[target_bsz:target_bsz * 2, :, img_s:img_e, :]
            ref_len = int(ref_k.shape[2])
            if ref_len <= 0:
                return flux_attention(q, k, v, pe=None, mask=attn_mask, transformer_options=transformer_options)

            q_t = q[:target_bsz]
            k_t = torch.cat([k[:target_bsz], ref_k], dim=2)
            v_t = torch.cat([v[:target_bsz], ref_v], dim=2)
            k_t, v_t = _flux_kv_heads_if_needed(k_t, v_t, int(q_heads))
            mask_t = _flux_append_ref_padding_to_mask(attn_mask, target_bsz, ref_len)

            out_t = optimized_attention_masked(
                q_t, k_t, v_t, int(q_heads), mask_t,
                skip_reshape=True, transformer_options=transformer_options,
            )

            q_r = q[target_bsz:target_bsz * 2]
            k_r = k[target_bsz:target_bsz * 2]
            v_r = v[target_bsz:target_bsz * 2]
            k_r, v_r = _flux_kv_heads_if_needed(k_r, v_r, int(q_heads))
            mask_r = _flux_slice_mask(attn_mask, target_bsz, target_bsz * 2)
            out_r = optimized_attention_masked(
                q_r, k_r, v_r, int(q_heads), mask_r,
                skip_reshape=True, transformer_options=transformer_options,
            )

            post_a = _coerce_strength01(cfg.get("post_attention_adain_strength", 0.0))
            if post_a > 0.0:
                out_t_adain = _adain(out_t, out_r, eps=1e-6)
                out_t = out_t * (1.0 - post_a) + out_t_adain * post_a

            outs = [out_t, out_r]
            if bsz > target_bsz * 2:
                q_e = q[target_bsz * 2:]
                k_e = k[target_bsz * 2:]
                v_e = v[target_bsz * 2:]
                k_e, v_e = _flux_kv_heads_if_needed(k_e, v_e, int(q_heads))
                mask_e = _flux_slice_mask(attn_mask, target_bsz * 2, bsz)
                outs.append(optimized_attention_masked(
                    q_e, k_e, v_e, int(q_heads), mask_e,
                    skip_reshape=True, transformer_options=transformer_options,
                ))

            return torch.cat(outs, dim=0)
        except Exception as exc:
            if hasattr(stats, "adapter_attn_failures"):
                stats.adapter_attn_failures += 1
            print(f"{prefix} ⚠ {DISPLAY_NAME} attention patch failed in {module_name}: {exc}")
            if _verbose_enabled():
                traceback.print_exc()
            return flux_attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)

    matched = installed = restored = 0
    patched_names: List[str] = []

    # Double-stream blocks: joint attention over [txt, img] happens inside the block.
    for idx, block in enumerate(getattr(dm, "double_blocks", []) or []):
        if not all(hasattr(block, attr) for attr in ("img_attn", "txt_attn", "img_norm1", "txt_norm1")):
            continue
        matched += 1
        name = f"double_blocks.{idx}"
        patched_names.append(name)

        if hasattr(block, "_untwist_orig_flux2_forward"):
            block.forward = block._untwist_orig_flux2_forward
            restored += 1
        else:
            block._untwist_orig_flux2_forward = block.forward
        original_forward = block._untwist_orig_flux2_forward

        def make_double_forward(orig, module_name):
            def patched_double_forward(
                self,
                img,
                txt,
                vec,
                pe,
                attn_mask=None,
                modulation_dims_img=None,
                modulation_dims_txt=None,
                transformer_options={},
            ):
                try:
                    if self.modulation:
                        img_mod1, img_mod2 = self.img_mod(vec)
                        txt_mod1, txt_mod2 = self.txt_mod(vec)
                    else:
                        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

                    transformer_patches = transformer_options.get("patches", {})
                    extra_options = transformer_options.copy()

                    img_modulated = self.img_norm1(img)
                    img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
                    img_qkv = self.img_attn.qkv(img_modulated)
                    del img_modulated
                    img_q, img_k, img_v = img_qkv.view(
                        img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1
                    ).permute(2, 0, 3, 1, 4)
                    del img_qkv
                    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

                    txt_modulated = self.txt_norm1(txt)
                    txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
                    txt_qkv = self.txt_attn.qkv(txt_modulated)
                    del txt_modulated
                    txt_q, txt_k, txt_v = txt_qkv.view(
                        txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1
                    ).permute(2, 0, 3, 1, 4)
                    del txt_qkv
                    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

                    q = torch.cat((txt_q, img_q), dim=2)
                    del txt_q, img_q
                    k = torch.cat((txt_k, img_k), dim=2)
                    del txt_k, img_k
                    v = torch.cat((txt_v, img_v), dim=2)
                    del txt_v, img_v

                    img_start = int(txt.shape[1])
                    img_end = int(q.shape[2])
                    extra_options["img_slice"] = [img_start, img_end]

                    if "attn1_patch" in transformer_patches:
                        for patch in transformer_patches["attn1_patch"]:
                            out = patch(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
                            q = out.get("q", q)
                            k = out.get("k", k)
                            v = out.get("v", v)
                            pe = out.get("pe", pe)
                            attn_mask = out.get("attn_mask", attn_mask)

                    attn = _reference_attention(
                        q, k, v, pe, attn_mask, transformer_options,
                        (img_start, img_end), module_name,
                    )
                    del q, k, v

                    if "attn1_output_patch" in transformer_patches:
                        for patch in transformer_patches["attn1_output_patch"]:
                            attn = patch(attn, extra_options)

                    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

                    img += apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
                    del img_attn
                    img += apply_mod(
                        self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)),
                        img_mod2.gate,
                        None,
                        modulation_dims_img,
                    )

                    txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
                    del txt_attn
                    txt += apply_mod(
                        self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)),
                        txt_mod2.gate,
                        None,
                        modulation_dims_txt,
                    )

                    if txt.dtype == torch.float16:
                        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

                    return img, txt
                except Exception as exc:
                    if hasattr(stats, "adapter_attn_failures"):
                        stats.adapter_attn_failures += 1
                    print(f"{prefix} ⚠ {DISPLAY_NAME} double block patch failed in {module_name}: {exc}")
                    if _verbose_enabled():
                        traceback.print_exc()
                    return orig(
                        img,
                        txt,
                        vec,
                        pe,
                        attn_mask=attn_mask,
                        modulation_dims_img=modulation_dims_img,
                        modulation_dims_txt=modulation_dims_txt,
                        transformer_options=transformer_options,
                    )
            return patched_double_forward

        block.forward = types.MethodType(make_double_forward(original_forward, name), block)
        setattr(block, "_untwist_flux2_active", True)
        installed += 1

    # Single-stream blocks: Q/K/V are produced inside the block, no child attn module.
    for idx, block in enumerate(getattr(dm, "single_blocks", []) or []):
        if not all(hasattr(block, attr) for attr in ("linear1", "linear2", "norm", "hidden_size", "num_heads")):
            continue
        matched += 1
        name = f"single_blocks.{idx}"
        patched_names.append(name)

        if hasattr(block, "_untwist_orig_flux2_forward"):
            block.forward = block._untwist_orig_flux2_forward
            restored += 1
        else:
            block._untwist_orig_flux2_forward = block.forward
        original_forward = block._untwist_orig_flux2_forward

        def make_single_forward(orig, module_name):
            def patched_single_forward(
                self,
                x,
                vec,
                pe,
                attn_mask=None,
                modulation_dims=None,
                transformer_options={},
            ):
                try:
                    if self.modulation:
                        mod, _ = self.modulation(vec)
                    else:
                        mod = vec

                    transformer_patches = transformer_options.get("patches", {})
                    extra_options = transformer_options.copy()

                    qkv, mlp = torch.split(
                        self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)),
                        [3 * self.hidden_size, self.mlp_hidden_dim_first],
                        dim=-1,
                    )
                    q, k, v = qkv.view(
                        qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1
                    ).permute(2, 0, 3, 1, 4)
                    del qkv
                    q, k = self.norm(q, k, v)

                    if "attn1_patch" in transformer_patches:
                        for patch in transformer_patches["attn1_patch"]:
                            out = patch(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
                            q = out.get("q", q)
                            k = out.get("k", k)
                            v = out.get("v", v)
                            pe = out.get("pe", pe)
                            attn_mask = out.get("attn_mask", attn_mask)

                    img_slice = transformer_options.get("img_slice", None)
                    if isinstance(img_slice, (list, tuple)) and len(img_slice) >= 2:
                        image_range = (int(img_slice[0]), int(img_slice[1]))
                    else:
                        image_range = (0, int(q.shape[2]))

                    attn = _reference_attention(
                        q, k, v, pe, attn_mask, transformer_options,
                        image_range, module_name,
                    )
                    del q, k, v

                    if "attn1_output_patch" in transformer_patches:
                        for patch in transformer_patches["attn1_output_patch"]:
                            attn = patch(attn, extra_options)

                    if self.yak_mlp:
                        mlp = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2:]) * mlp[..., :self.mlp_hidden_dim_first // 2]
                    else:
                        mlp = self.mlp_act(mlp)

                    output = self.linear2(torch.cat((attn, mlp), 2))
                    x += apply_mod(output, mod.gate, None, modulation_dims)

                    if x.dtype == torch.float16:
                        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)

                    return x
                except Exception as exc:
                    if hasattr(stats, "adapter_attn_failures"):
                        stats.adapter_attn_failures += 1
                    print(f"{prefix} ⚠ {DISPLAY_NAME} single block patch failed in {module_name}: {exc}")
                    if _verbose_enabled():
                        traceback.print_exc()
                    return orig(
                        x,
                        vec,
                        pe,
                        attn_mask=attn_mask,
                        modulation_dims=modulation_dims,
                        transformer_options=transformer_options,
                    )
            return patched_single_forward

        block.forward = types.MethodType(make_single_forward(original_forward, name), block)
        setattr(block, "_untwist_flux2_active", True)
        installed += 1

    return matched, installed, restored, patched_names


def uses_reference_branch_kv() -> bool:
    return False


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def describe_match(model_info: dict[str, Any]) -> str:
    model_config_class = str(model_info.get("model_config_class", ""))
    unet_config = model_info.get("unet_config", {})
    image_model = ""
    if isinstance(unet_config, dict):
        image_model = str(unet_config.get("image_model", ""))
    else:
        image_model = str(model_info.get("image_model", ""))

    supported = ", ".join(sorted(SUPPORTED_MODEL_CONFIG_CLASSES))
    return (
        f"{DISPLAY_NAME}: model_config_class={model_config_class!r}, "
        f"image_model={image_model!r}, supported_classes={{{supported}}}"
    )


__all__ = [
    "ARCHITECTURE",
    "DISPLAY_NAME",
    "SUPPORTED_MODEL_CONFIG_CLASSES",
    "matches_model",
    "is_model_identity",
    "find_diffusion_model",
    "default_runtime_cfg",
    "axes_dims_from_dm",
    "head_dim_from_dm",
    "is_attention_name",
    "is_double_stream_attention_name",
    "is_single_stream_block_name",
    "block_index_from_name",
    "stream_kind_from_name",
    "is_flux2_self_attention_module",
    "is_flux2_single_stream_block",
    "is_joint_attention",
    "iter_flux2_patch_targets",
    "prepare_reference_conditioning",
    "patch_attention_modules",
    "uses_reference_branch_kv",
    "describe_match",
]
