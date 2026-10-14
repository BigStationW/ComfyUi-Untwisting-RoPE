from __future__ import annotations

"""Krea 2 adapter for ComfyUi-Untwisting-RoPE.

Single-stream MMDiT: fused text+image tokens processed by blocks.N.attn.
Style transfer mirrors other adapters: target Q attends to target K/V plus
scaled reference-image K/V; shared QKV and attention-output effects are
delegated to the top-level UntwistingRoPE helpers.

imglen injection: dm.txtmlp.forward is patched to store txtlen on dm after
the text fusion step. The block-level patch reads this and injects
krea2_imglen = seqlen - txtlen into transformer_options before self.attn
is called. This is reliable because txtmlp is a plain nn.Module whose
forward survives VRAM reloads, unlike dm._forward which is bypassed by
WrapperExecutor.
"""

import math
import types
from typing import Any, List, Optional

import torch
from einops import rearrange
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked


# ---------------------------------------------------------------------------
# Adapter identity
# ---------------------------------------------------------------------------

ARCHITECTURE = "krea2"
DISPLAY_NAME = "Krea 2"
CONFIG_KEY = "untwisting_rope"

SUPPORTED_MODEL_CONFIG_CLASSES: set[str] = {"Krea2"}

DIFFUSION_ATTR_PATHS = (
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
    "diffusion_model",
)


# ---------------------------------------------------------------------------
# Generic helpers
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


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


# ---------------------------------------------------------------------------
# Required adapter hooks
# ---------------------------------------------------------------------------

def matches_model(model_info: dict[str, Any]) -> bool:
    return str(model_info.get("model_config_class", "")) in SUPPORTED_MODEL_CONFIG_CLASSES


def is_model_identity(model_info: dict[str, Any]) -> bool:
    return matches_model(model_info)


def find_diffusion_model(model_patcher: Any) -> Any:
    for path in DIFFUSION_ATTR_PATHS:
        obj, ok = _get_attr_path(model_patcher, path)
        if ok and obj is not None:
            return obj
    raise RuntimeError(f"Could not find ComfyUI BaseModel.diffusion_model for {DISPLAY_NAME}.")


# ---------------------------------------------------------------------------
# Krea 2 metadata
# ---------------------------------------------------------------------------

def axes_dims_from_head_dim(head_dim: int) -> List[int]:
    """Exact 3-axis split from comfy/ldm/krea2/model.py SingleStreamDiT."""
    hd = int(head_dim)
    axes = [hd - 12 * (hd // 16), 6 * (hd // 16), 6 * (hd // 16)]
    if sum(axes) != hd or any(v <= 0 for v in axes):
        raise RuntimeError(
            f"{DISPLAY_NAME} axes_dims: cannot split head_dim={hd} into [T,H,W]: {axes}."
        )
    return axes


def _first_main_attention(dm: Any) -> Any:
    blocks = getattr(dm, "blocks", None)
    if blocks is None:
        raise RuntimeError(f"{DISPLAY_NAME} metadata lookup failed: dm.blocks is missing.")
    try:
        block0 = blocks[0]
    except Exception as exc:
        raise RuntimeError(f"{DISPLAY_NAME} dm.blocks[0] unavailable.") from exc
    attn = getattr(block0, "attn", None)
    if attn is None:
        raise RuntimeError(f"{DISPLAY_NAME} dm.blocks[0].attn is missing.")
    return attn


def head_dim_from_dm(dm: Any | None) -> int:
    if dm is None:
        return 128  # Krea2 default: features=6144, heads=48
    attn = _first_main_attention(dm)
    head_dim = _safe_int(getattr(attn, "headdim", None))
    if head_dim is None or head_dim <= 0:
        raise RuntimeError(f"{DISPLAY_NAME} invalid headdim={head_dim!r}.")
    return int(head_dim)


def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    head_dim = head_dim_from_dm(dm)
    return {
        "architecture": ARCHITECTURE,
        "head_dim": head_dim,
        "axes_dims": axes_dims_from_head_dim(head_dim),
        "target_qk_adain_ranges": [(0, 2**31 - 1)],
    }


# ---------------------------------------------------------------------------
# Patch target predicates
# ---------------------------------------------------------------------------

def _index_in_range(parts: list[str], min_layer: int, max_layer: int) -> bool:
    if len(parts) < 2:
        return False
    idx = _safe_int(parts[1])
    if idx is None:
        return False
    return int(min_layer) <= idx <= int(max_layer)


def is_main_block_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    parts = str(name).split(".")
    return len(parts) == 2 and parts[0] == "blocks" and _index_in_range(parts, min_layer, max_layer)


def is_main_block_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    parts = str(name).split(".")
    return (
        len(parts) == 3
        and parts[0] == "blocks"
        and parts[2] == "attn"
        and _index_in_range(parts, min_layer, max_layer)
    )


def is_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    return is_main_block_attention_name(name, min_layer, max_layer)


def block_index_from_name(name: str) -> int:
    parts = str(name).split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        idx = _safe_int(parts[1], -1)
        return -1 if idx is None else int(idx)
    return -1


def is_krea2_attention_module(module: Any) -> bool:
    required = ("wq", "wk", "wv", "wo", "gate", "qknorm", "heads", "kvheads", "headdim", "forward")
    return all(hasattr(module, attr) for attr in required) and callable(getattr(module, "forward", None))


def is_krea2_single_stream_block(module: Any) -> bool:
    required = ("mod", "prenorm", "postnorm", "attn", "mlp", "forward")
    return all(hasattr(module, attr) for attr in required) and callable(getattr(module, "forward", None))


def is_joint_attention(module: Any) -> bool:
    return is_krea2_attention_module(module)


def prepare_reference_conditioning(
    ref_conditioning: Any,
    dm: Any,
    device: Any,
    dtype: Any,
    stats: Any = None,
    label: str = "",
    helpers: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    return ref_conditioning, "not-applicable"


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _expand_kv_heads(
    k: torch.Tensor, v: torch.Tensor, q_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_heads = int(k.shape[1])
    q_heads = int(q_heads)
    if kv_heads == q_heads:
        return k, v
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise RuntimeError(
            f"{DISPLAY_NAME} cannot expand KV heads: q_heads={q_heads}, kv_heads={kv_heads}."
        )
    rep = q_heads // kv_heads
    return k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)


def _image_range(transformer_options: Any, dm: Any, seqlen: int) -> tuple[int, int]:
    """Resolve image token range from transformer_options or dm fallback."""
    imglen = None
    if isinstance(transformer_options, dict):
        imglen = transformer_options.get("krea2_imglen", None)
    if imglen is None:
        imglen = getattr(dm, "_untwist_krea2_last_imglen", None)
    if imglen is None:
        raise RuntimeError(
            f"{DISPLAY_NAME} could not determine image token length. "
            "txtmlp hook did not populate dm._untwist_krea2_last_txtlen."
        )
    imglen_i = max(0, min(int(imglen), int(seqlen)))
    img_s = int(seqlen) - imglen_i
    img_e = int(seqlen)
    if img_e <= img_s:
        raise RuntimeError(
            f"{DISPLAY_NAME} empty image token range: imglen={imglen_i}, seqlen={seqlen}."
        )
    return img_s, img_e


# ---------------------------------------------------------------------------
# Main patch
# ---------------------------------------------------------------------------

def patch_attention_modules(
    dm: Any,
    stats: Any,
    helpers: dict[str, Any] | None = None,
) -> tuple[int, int, int, list[str]]:
    helpers = helpers or {}
    prefix = str(helpers.get("prefix", "[UntwistingRoPE]"))
    config_key = str(helpers.get("config_key", CONFIG_KEY))

    required_helpers = (
        "patch_context_refiner_mask_modules",
        "build_frequency_scale_vector",
        "apply_qkv_shared_effects",
        "apply_attention_output_shared_effects",
    )
    missing = [n for n in required_helpers if not callable(helpers.get(n))]
    if missing:
        raise RuntimeError(f"{DISPLAY_NAME} adapter missing required helper(s): {missing}")

    build_frequency_scale_vector = helpers["build_frequency_scale_vector"]
    apply_qkv_shared_effects = helpers["apply_qkv_shared_effects"]
    apply_attention_output_shared_effects = helpers["apply_attention_output_shared_effects"]

    helpers["patch_context_refiner_mask_modules"](dm, stats)

    # ------------------------------------------------------------------
    # imglen injection strategy
    # ------------------------------------------------------------------
    # dm._forward is bypassed by WrapperExecutor so we cannot rely on it.
    # Instead we patch two plain nn.Module submodules whose forwards DO
    # survive VRAM reloads:
    #
    # 1. dm.txtmlp.forward: runs right before combined = cat(context, img)
    #    in _forward. Its input x has shape [B, txtlen, features].
    #    We store txtlen on dm.
    #
    # 2. dm.blocks[N].forward (SingleStreamBlock): runs in the block loop
    #    with x = combined [B, seqlen, C]. We compute imglen = seqlen -
    #    txtlen and inject krea2_imglen into transformer_options before
    #    self.attn is called.
    # ------------------------------------------------------------------

    # Patch txtmlp to store txtlen.
    txtmlp = getattr(dm, "txtmlp", None)
    if txtmlp is not None and not hasattr(txtmlp, "_untwist_orig_krea2_txtmlp_forward"):
        txtmlp._untwist_orig_krea2_txtmlp_forward = txtmlp.forward
        orig_txtmlp_fwd = txtmlp._untwist_orig_krea2_txtmlp_forward

        def patched_txtmlp_forward(self, x):
            dm._untwist_krea2_last_txtlen = (
                int(x.shape[1]) if torch.is_tensor(x) and x.ndim >= 2 else None
            )
            return orig_txtmlp_fwd(x)

        txtmlp.forward = types.MethodType(patched_txtmlp_forward, txtmlp)

    # Patch SingleStreamBlock.forward to inject krea2_imglen.
    for name, module in dm.named_modules():
        if not is_main_block_name(name, 0, 999):
            continue
        if not is_krea2_single_stream_block(module):
            continue
        if hasattr(module, "_untwist_orig_krea2_block_forward"):
            continue

        module._untwist_orig_krea2_block_forward = module.forward
        orig_block_fwd = module._untwist_orig_krea2_block_forward

        def make_block_forward(orig):
            def patched_block_forward(self, x, vec, freqs, mask=None, transformer_options={}):
                if isinstance(transformer_options, dict) and "krea2_imglen" not in transformer_options:
                    cfg = transformer_options.get(config_key)
                    if cfg and cfg.get("enabled"):
                        txtlen = getattr(dm, "_untwist_krea2_last_txtlen", None)
                        if txtlen is not None and torch.is_tensor(x) and x.ndim >= 2:
                            imglen = x.shape[1] - txtlen
                            if imglen > 0:
                                transformer_options = transformer_options.copy()
                                transformer_options["krea2_imglen"] = imglen
                return orig(x, vec, freqs, mask=mask, transformer_options=transformer_options)
            return patched_block_forward

        module.forward = types.MethodType(make_block_forward(orig_block_fwd), module)

    # ------------------------------------------------------------------
    # Attention patch
    # ------------------------------------------------------------------
    matched = installed = restored = 0
    patched_names: list[str] = []

    for name, module in dm.named_modules():
        if not is_attention_name(name, 0, 999):
            continue
        if not is_krea2_attention_module(module):
            continue

        matched += 1
        patched_names.append(name)

        if hasattr(module, "_untwist_orig_krea2_attention_forward"):
            module.forward = module._untwist_orig_krea2_attention_forward
            restored += 1
        else:
            module._untwist_orig_krea2_attention_forward = module.forward

        original_forward = module._untwist_orig_krea2_attention_forward

        def make_forward(orig, module_name: str):
            def patched_forward(self, x, freqs=None, mask=None, transformer_options={}):
                try:
                    cfg = (
                        transformer_options.get(config_key)
                        if isinstance(transformer_options, dict) else None
                    )
                    if not cfg or not cfg.get("enabled"):
                        return orig(x, freqs=freqs, mask=mask,
                                    transformer_options=transformer_options)

                    target_bsz = int(cfg.get("cross_batch_target_batch", 0))
                    if target_bsz <= 0:
                        raise RuntimeError(
                            f"{DISPLAY_NAME} enabled in {module_name} but "
                            f"cross_batch_target_batch={target_bsz}."
                        )
                    if not torch.is_tensor(x) or x.ndim != 3:
                        raise RuntimeError(
                            f"{DISPLAY_NAME} expected x as [B,S,C] in {module_name}; "
                            f"got {type(x).__name__} ndim={getattr(x, 'ndim', None)}."
                        )

                    bsz, seqlen, _ = x.shape
                    if bsz < target_bsz * 2:
                        raise RuntimeError(
                            f"{DISPLAY_NAME} expected target+reference batches in "
                            f"{module_name}; bsz={bsz}, target_bsz={target_bsz}."
                        )

                    block_idx = int(transformer_options.get(
                        "block_index", block_index_from_name(module_name)
                    ))
                    active_blocks = cfg.get("active_blocks", set())
                    if active_blocks and block_idx not in active_blocks:
                        return orig(x, freqs=freqs, mask=mask,
                                    transformer_options=transformer_options)

                    img_s, img_e = _image_range(transformer_options, dm, seqlen)
                    token_ranges = [(img_s, img_e)]

                    if hasattr(stats, "attn_calls"):
                        stats.attn_calls += 1

                    q_heads = int(getattr(self, "heads", 0))
                    kv_heads = int(getattr(self, "kvheads", q_heads))
                    head_dim = int(getattr(self, "headdim", 0))
                    if q_heads <= 0 or kv_heads <= 0 or head_dim <= 0:
                        raise RuntimeError(
                            f"{DISPLAY_NAME} invalid attention metadata in {module_name}: "
                            f"heads={q_heads}, kvheads={kv_heads}, headdim={head_dim}."
                        )

                    # Q/K/V projections — BHSD layout throughout.
                    q = rearrange(self.wq(x), "B L (H D) -> B H L D", H=q_heads)
                    k = rearrange(self.wk(x), "B L (H D) -> B H L D", H=kv_heads)
                    v = rearrange(self.wv(x), "B L (H D) -> B H L D", H=kv_heads)
                    gate = self.gate(x)

                    q, k = self.qknorm(q, k)
                    if freqs is not None:
                        q, k = apply_rope(q, k, freqs)

                    # GQA expand before shared effects so helpers see full heads.
                    k, v = _expand_kv_heads(k, v, q_heads)

                    # Shared AdaIN/cosine effects on image tokens, BHSD layout.
                    q, k, v = apply_qkv_shared_effects(
                        q, k, v,
                        cfg,
                        target_bsz,
                        module_name,
                        layout="BHSD",
                        token_ranges=token_ranges,
                    )

                    # Frequency scale vector for reference K modulation.
                    progress = float(cfg.get("progress", 0.0))
                    high_scale = _lerp(
                        float(cfg["high_scale_start"]), float(cfg["high_scale_end"]), progress
                    )
                    low_scale = _lerp(
                        float(cfg["low_scale_start"]), float(cfg["low_scale_end"]), progress
                    )
                    beta = float(cfg.get("beta", 2.0))
                    axes_dims = cfg.get("axes_dims") or axes_dims_from_head_dim(head_dim)
                    scale_vec = build_frequency_scale_vector(
                        head_dim, axes_dims, high_scale, low_scale, beta,
                        k.device, k.dtype, runtime_cfg=cfg,
                    ).view(1, 1, 1, head_dim)

                    # Target attends to its own K/V plus scaled reference image K/V.
                    # V is left unscaled — only K frequency content is modulated.
                    ref_k_img = k[target_bsz:target_bsz * 2, :, img_s:img_e, :] * scale_vec
                    ref_v_img = v[target_bsz:target_bsz * 2, :, img_s:img_e, :]

                    k_t = torch.cat([k[:target_bsz], ref_k_img], dim=2)
                    v_t = torch.cat([v[:target_bsz], ref_v_img], dim=2)

                    # If a mask is supplied (unexpected but guard for future compat),
                    # fall back to native attention to avoid shape mismatch.
                    if mask is not None:
                        return orig(x, freqs=freqs, mask=mask,
                                    transformer_options=transformer_options)

                    out_t = optimized_attention_masked(
                        q[:target_bsz], k_t, v_t, q_heads,
                        mask=None, skip_reshape=True,
                        transformer_options=transformer_options,
                    )
                    out_r = optimized_attention_masked(
                        q[target_bsz:target_bsz * 2],
                        k[target_bsz:target_bsz * 2],
                        v[target_bsz:target_bsz * 2],
                        q_heads, mask=None, skip_reshape=True,
                        transformer_options=transformer_options,
                    )

                    out_t, out_r = apply_attention_output_shared_effects(
                        out_t, out_r, cfg, target_bsz, module_name,
                        layout="BSD", token_ranges=token_ranges,
                    )

                    outs = [out_t, out_r]
                    if bsz > target_bsz * 2:
                        out_extra = optimized_attention_masked(
                            q[target_bsz * 2:],
                            k[target_bsz * 2:],
                            v[target_bsz * 2:],
                            q_heads, mask=None, skip_reshape=True,
                            transformer_options=transformer_options,
                        )
                        outs.append(out_extra)

                    out = torch.cat(outs, dim=0)
                    if out.shape != gate.shape:
                        raise RuntimeError(
                            f"{DISPLAY_NAME} output/gate shape mismatch in {module_name}: "
                            f"out={tuple(out.shape)}, gate={tuple(gate.shape)}."
                        )
                    return self.wo(out * torch.sigmoid(gate))

                except Exception as exc:
                    raise RuntimeError(
                        f"{DISPLAY_NAME} attention patch failed in {module_name}: {exc}"
                    ) from exc

            return patched_forward

        module.forward = types.MethodType(make_forward(original_forward, name), module)
        installed += 1

    if installed <= 0:
        raise RuntimeError(
            f"{DISPLAY_NAME} adapter patch failed: no compatible blocks.N.attn modules found."
        )
    return matched, installed, restored, patched_names


def uses_reference_branch_kv() -> bool:
    return False


def describe_match(model_info: dict[str, Any]) -> str:
    model_config_class = str(model_info.get("model_config_class", ""))
    supported = ", ".join(sorted(SUPPORTED_MODEL_CONFIG_CLASSES))
    return (
        f"{DISPLAY_NAME}: model_config_class={model_config_class!r}, "
        f"supported_classes={{{supported}}}"
    )


__all__ = [
    "ARCHITECTURE",
    "DISPLAY_NAME",
    "CONFIG_KEY",
    "SUPPORTED_MODEL_CONFIG_CLASSES",
    "matches_model",
    "is_model_identity",
    "find_diffusion_model",
    "default_runtime_cfg",
    "axes_dims_from_head_dim",
    "is_attention_name",
    "is_main_block_attention_name",
    "block_index_from_name",
    "is_krea2_attention_module",
    "is_joint_attention",
    "prepare_reference_conditioning",
    "patch_attention_modules",
    "uses_reference_branch_kv",
    "describe_match",
]