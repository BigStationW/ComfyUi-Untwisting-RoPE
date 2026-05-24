from __future__ import annotations
import math
import types
import hashlib
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple
from . import models as model_adapters
import torch
import comfy.utils
import comfy.patcher_extension
try:
    import latent_preview
except Exception:
    latent_preview = None
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked

_PREFIX = '[UntwistingRoPE]'
_TRANSFORMER_CONFIG_KEY = model_adapters.CONFIG_KEY

# Module-level fallback store. Comfy/KSampler may clone or pass model objects
# through different instances, so the export node reads this if the model-local
_RF_LAST_DEBUG_STORE: Dict[str, Any] = {
    'cache': {},
    'sampler_sigmas': None,
    'built_sigmas': None,
    'run_count': 0,
}

# Persistent RF trajectory cache shared across prompt executions.
# Keyed by reference latent, reference conditioning, sigma schedule, RF mode,
# and RF parameters. Values are stored on CPU to avoid pinning VRAM.
_RF_PERSISTENT_TRAJECTORY_CACHE: Dict[str, Dict[str, Any]] = {}
_RF_PERSISTENT_CACHE_MAX_ITEMS = 4

# ═══════════════════════════════════════════════════════════════════════════════
# Persistent RF cache helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_update_tensor(h: "hashlib._Hash", t: torch.Tensor, full: bool = True) -> None:
    td = t.detach().to(device='cpu').contiguous()
    h.update(str(tuple(td.shape)).encode('utf-8'))
    h.update(str(td.dtype).encode('utf-8'))
    if full:
        h.update(td.numpy().tobytes())
    else:
        flat = td.flatten()
        if flat.numel() > 4096:
            flat = flat[torch.linspace(0, flat.numel() - 1, 4096).long()]
        h.update(flat.numpy().tobytes())

def _hash_any(obj: Any, h: Optional["hashlib._Hash"] = None, depth: int = 0) -> str:
    if h is None:
        h = hashlib.sha1()
    if depth > 12:
        h.update(b'<maxdepth>')
        return h.hexdigest()
    if torch.is_tensor(obj):
        h.update(b'TENSOR')
        _hash_update_tensor(h, obj, full=True)
    elif isinstance(obj, dict):
        h.update(b'DICT')
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            if str(k) == 'transformer_options':
                continue
            h.update(str(k).encode('utf-8'))
            _hash_any(obj[k], h, depth + 1)
    elif isinstance(obj, (list, tuple)):
        h.update(b'LIST' if isinstance(obj, list) else b'TUPLE')
        h.update(str(len(obj)).encode('utf-8'))
        for v in obj:
            _hash_any(v, h, depth + 1)
    elif obj is None:
        h.update(b'NONE')
    else:
        h.update(repr(obj).encode('utf-8', errors='ignore'))
    return h.hexdigest()

def _make_rf_persistent_key(
    ref_clean: torch.Tensor,
    ref_conditioning: Any,
    sampler_sigmas: List[float],
    target_b: int,
    rf_mode: str,
    gamma: float,
    gamma_curve: float,
    norm_strength: float,
    cond_mode: str,
    pmi_alpha: float = 0.4,
) -> str:
    h = hashlib.sha1()

    h.update(str(tuple(ref_clean.shape)).encode('utf-8'))
    _hash_update_tensor(h, ref_clean, full=True)

    h.update(_hash_any(ref_conditioning).encode('utf-8'))

    h.update(str([round(float(s), 8) for s in sampler_sigmas]).encode('utf-8'))
    h.update(str(int(target_b)).encode('utf-8'))

    h.update(str(rf_mode).encode('utf-8'))
    h.update(f'{float(gamma):.8f}'.encode('utf-8'))
    h.update(f'{float(gamma_curve):.8f}'.encode('utf-8'))
    h.update(f'{float(norm_strength):.8f}'.encode('utf-8'))
    h.update(str(cond_mode).encode('utf-8'))
    h.update(f'{float(pmi_alpha):.8f}'.encode('utf-8'))

    return h.hexdigest()

def _cache_to_cpu(cache: Dict[float, torch.Tensor]) -> Dict[float, torch.Tensor]:
    return {
        float(k): v.detach().to(device='cpu').clone()
        for k, v in cache.items()
        if torch.is_tensor(v)
    }

def _cache_to_device(cache: Dict[float, torch.Tensor], device, dtype) -> Dict[float, torch.Tensor]:
    return {
        float(k): v.to(device=device, dtype=dtype).detach().clone()
        for k, v in cache.items()
        if torch.is_tensor(v)
    }

def _put_persistent_rf_cache(key: str, entry: Dict[str, Any]) -> None:
    _RF_PERSISTENT_TRAJECTORY_CACHE[key] = entry
    while len(_RF_PERSISTENT_TRAJECTORY_CACHE) > _RF_PERSISTENT_CACHE_MAX_ITEMS:
        oldest = next(iter(_RF_PERSISTENT_TRAJECTORY_CACHE.keys()))
        _RF_PERSISTENT_TRAJECTORY_CACHE.pop(oldest, None)

def _coerce_bool(value: Any) -> bool:
    """Robust boolean parsing for ComfyUI values that may arrive as bools or strings."""
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on', 'y', 't')
    return bool(value)

# ═══════════════════════════════════════════════════════════════════════════════
# Runtime state
# ═══════════════════════════════════════════════════════════════════════════════

class _RuntimeStats:
    def __init__(self, verbose: bool = False, rf_verbose: bool = False) -> None:
        # verbose controls UntwistingRoPE patch/attention logs.
        # rf_verbose controls RFInversion trajectory/wrapper logs.
        self.verbose: bool = _coerce_bool(verbose)
        self.rf_verbose: bool = _coerce_bool(rf_verbose)
        self.wrapper_calls:  int = 0
        self.patchify_calls: int = 0
        self.attn_calls:     int = 0
        self.context_refiner_calls: int = 0
        self.adapter_attn_calls: int = 0
        self.adapter_attn_failures: int = 0

        self.rf_sigma_cache: Dict[float, torch.Tensor] = {}
        self.rf_eps: Optional[torch.Tensor] = None
        self.rf_prev_z: Optional[torch.Tensor] = None
        self.rf_prev_sigma: Optional[float] = None
        self.rf_step_count: int = 0
        self.rf_run_count: int = 0
        self.rf_sampler_sigmas: Optional[List[float]] = None
        self.rf_schedule_built: bool = False

        self.fixed_noise: Optional[torch.Tensor] = None

        self.scale_vec_logged:  bool = False
        self.joint_mask_logged: bool = False

        # Parameterization detection: tracks whether apply_model is x0 or velocity
        self.parameterization: str = 'unknown'
        

def _vprint(stats: Optional[_RuntimeStats], *args, **kwargs) -> None:
    if stats is not None and _coerce_bool(getattr(stats, 'verbose', False)):
        print(*args, **kwargs)


def _rf_vprint(stats: Optional[_RuntimeStats], *args, **kwargs) -> None:
    if stats is not None and _coerce_bool(getattr(stats, 'rf_verbose', False)):
        print(*args, **kwargs)


def _rf_tensor_summary(name: str, value: Any) -> str:
    """Compact tensor diagnostic string safe for dtype/device/empty tensors."""
    if not torch.is_tensor(value):
        return f'{name}=<{type(value).__name__}>'
    try:
        shape = tuple(int(v) for v in value.shape)
        base = f'{name}: shape={shape} dtype={value.dtype} device={value.device}'
        if value.numel() == 0:
            return base + ' empty'
        vf = value.detach().float()
        return (
            f'{base} mean={float(vf.mean().item()):.6f} '
            f'std={float(vf.std(unbiased=False).item()):.6f} '
            f'min={float(vf.min().item()):.6f} max={float(vf.max().item()):.6f}'
        )
    except Exception as exc:
        return f'{name}: tensor-summary-failed shape={tuple(value.shape)} err={exc}'


def _rf_sequence_summary(name: str, seq: Any, max_items: int = 8) -> str:
    values = _coerce_sigma_sequence(seq)
    if values is None:
        return f'{name}=<none/invalid>'
    head = ', '.join(f'{v:.6f}' for v in values[:max_items])
    tail = ', '.join(f'{v:.6f}' for v in values[-max_items:])
    if len(values) <= max_items * 2:
        body = ', '.join(f'{v:.6f}' for v in values)
    else:
        body = f'{head}, ..., {tail}'
    return f'{name}: count={len(values)} min={min(values):.6f} max={max(values):.6f} values=[{body}]'


def _rf_brief_obj(obj: Any, depth: int = 0) -> str:
    """Small structural summary for conditioning/debug dictionaries."""
    if torch.is_tensor(obj):
        return f'Tensor{tuple(obj.shape)}:{obj.dtype}:{obj.device}'
    if obj is None:
        return 'None'
    if depth >= 2:
        return type(obj).__name__
    if isinstance(obj, dict):
        items = []
        for idx, (k, v) in enumerate(obj.items()):
            if idx >= 12:
                items.append('...')
                break
            items.append(f'{k}={_rf_brief_obj(v, depth + 1)}')
        return '{' + ', '.join(items) + '}'
    if isinstance(obj, (list, tuple)):
        items = []
        for idx, v in enumerate(obj):
            if idx >= 8:
                items.append('...')
                break
            items.append(_rf_brief_obj(v, depth + 1))
        return f'{type(obj).__name__}[{len(obj)}](' + ', '.join(items) + ')'
    return f'{type(obj).__name__}({repr(obj)[:80]})'



def _rf_tensor_stats(value: Any) -> Dict[str, Any]:
    """Numerical health stats used only for RF diagnostics."""
    out: Dict[str, Any] = {
        'is_tensor': torch.is_tensor(value),
        'finite': False,
        'numel': 0,
        'nan_count': None,
        'inf_count': None,
        'mean': None,
        'std': None,
        'min': None,
        'max': None,
        'max_abs': None,
    }
    if not torch.is_tensor(value):
        return out
    try:
        x = value.detach().float()
        out['numel'] = int(x.numel())
        if x.numel() == 0:
            out['finite'] = True
            return out
        finite = torch.isfinite(x)
        out['finite'] = bool(finite.all().item())
        out['nan_count'] = int(torch.isnan(x).sum().item())
        out['inf_count'] = int(torch.isinf(x).sum().item())
        xf = x[finite]
        if xf.numel() == 0:
            return out
        out['mean'] = float(xf.mean().item())
        out['std'] = float(xf.std(unbiased=False).item())
        out['min'] = float(xf.min().item())
        out['max'] = float(xf.max().item())
        out['max_abs'] = float(xf.abs().max().item())
    except Exception as exc:
        out['error'] = repr(exc)
    return out


def _rf_scalar_fmt(value: Any, digits: int = 6) -> str:
    try:
        if value is None:
            return 'n/a'
        value = float(value)
        if not math.isfinite(value):
            return str(value)
        return f'{value:.{digits}f}'
    except Exception:
        return 'n/a'


def _rf_tensor_mae(a: Any, b: Any) -> Optional[float]:
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return None
    try:
        aa = a.detach().float().to(device='cpu')
        bb = b.detach().float().to(device='cpu')
        if aa.shape != bb.shape or aa.numel() == 0:
            return None
        return float((aa - bb).abs().mean().item())
    except Exception:
        return None


def _rf_tensor_rmse(a: Any, b: Any) -> Optional[float]:
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return None
    try:
        aa = a.detach().float().to(device='cpu')
        bb = b.detach().float().to(device='cpu')
        if aa.shape != bb.shape or aa.numel() == 0:
            return None
        return float((aa - bb).pow(2).mean().sqrt().item())
    except Exception:
        return None


def _rf_tensor_cosine(a: Any, b: Any) -> Optional[float]:
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return None
    try:
        aa = a.detach().float().flatten().to(device='cpu')
        bb = b.detach().float().flatten().to(device='cpu')
        if aa.shape != bb.shape or aa.numel() == 0:
            return None
        denom = aa.norm() * bb.norm()
        if float(denom.item()) <= 1e-12:
            return None
        return float(torch.dot(aa, bb).div(denom).item())
    except Exception:
        return None


def _rf_diagnostic_level(summary: Dict[str, Any]) -> str:
    """Return a conservative numerical-health classification for the RF trajectory."""
    if not summary.get('finite_all', False):
        return 'FAIL'
    final_std = summary.get('final_std')
    final_max_abs = summary.get('final_max_abs')
    final_eps_std_ratio = summary.get('final_eps_std_ratio')

    # These thresholds are intentionally broad and only detect numerical collapse/divergence,
    # not subjective image quality or semantic faithfulness.
    try:
        if final_std is not None and float(final_std) < 0.05:
            return 'FAIL'
        if final_std is not None and float(final_std) > 20.0:
            return 'FAIL'
        if final_max_abs is not None and float(final_max_abs) > 100.0:
            return 'FAIL'
        if final_eps_std_ratio is not None and (
            float(final_eps_std_ratio) < 0.25 or float(final_eps_std_ratio) > 4.0
        ):
            return 'WARN'
    except Exception:
        return 'WARN'
    return 'PASS'


def _rf_stability_summary(
    ref_clean: torch.Tensor,
    eps: torch.Tensor,
    cache: Dict[float, torch.Tensor],
    sigmas: List[float],
) -> Dict[str, Any]:
    """Collect explicit RF trajectory sanity metrics without changing sampling."""
    keys = sorted(float(k) for k in cache.keys() if isinstance(k, (int, float)))
    summary: Dict[str, Any] = {
        'cache_items': len(cache),
        'sigmas': len(sigmas or []),
        'first_sigma': keys[0] if keys else None,
        'last_sigma': keys[-1] if keys else None,
        'finite_all': True,
        'level': 'WARN',
        'warnings': [],
    }
    if not keys:
        summary['finite_all'] = False
        summary['warnings'].append('cache_empty')
        summary['level'] = 'FAIL'
        return summary

    stds: List[float] = []
    max_abs_values: List[float] = []
    bad_keys: List[float] = []
    prev_tensor: Optional[torch.Tensor] = None
    dz_values: List[float] = []

    for k in keys:
        t = cache.get(k)
        st = _rf_tensor_stats(t)
        if not st.get('finite', False):
            bad_keys.append(k)
            summary['finite_all'] = False
        if st.get('std') is not None:
            stds.append(float(st['std']))
        if st.get('max_abs') is not None:
            max_abs_values.append(float(st['max_abs']))
        if torch.is_tensor(prev_tensor) and torch.is_tensor(t) and prev_tensor.shape == t.shape:
            try:
                dz_values.append(float((t.detach().float() - prev_tensor.detach().float()).abs().mean().item()))
            except Exception:
                pass
        prev_tensor = t if torch.is_tensor(t) else None

    first = cache.get(keys[0])
    final = cache.get(keys[-1])
    first_stats = _rf_tensor_stats(first)
    final_stats = _rf_tensor_stats(final)
    eps_stats = _rf_tensor_stats(eps)
    ref_stats = _rf_tensor_stats(ref_clean)

    summary.update({
        'bad_sigma_keys': bad_keys[:16],
        'std_min': min(stds) if stds else None,
        'std_max': max(stds) if stds else None,
        'max_abs_max': max(max_abs_values) if max_abs_values else None,
        'dz_mean': (sum(dz_values) / len(dz_values)) if dz_values else None,
        'dz_max': max(dz_values) if dz_values else None,
        'ref_std': ref_stats.get('std'),
        'eps_std': eps_stats.get('std'),
        'first_std': first_stats.get('std'),
        'final_std': final_stats.get('std'),
        'final_mean': final_stats.get('mean'),
        'final_min': final_stats.get('min'),
        'final_max': final_stats.get('max'),
        'final_max_abs': final_stats.get('max_abs'),
        'first_vs_ref_mae': _rf_tensor_mae(first, ref_clean),
        'final_vs_eps_mae': _rf_tensor_mae(final, eps),
        'final_vs_eps_rmse': _rf_tensor_rmse(final, eps),
        'final_vs_eps_cosine': _rf_tensor_cosine(final, eps),
    })

    try:
        eps_std = summary.get('eps_std')
        final_std = summary.get('final_std')
        if eps_std is not None and float(eps_std) > 1e-12 and final_std is not None:
            summary['final_eps_std_ratio'] = float(final_std) / float(eps_std)
        else:
            summary['final_eps_std_ratio'] = None
    except Exception:
        summary['final_eps_std_ratio'] = None

    if bad_keys:
        summary['warnings'].append('nonfinite_values')
    if summary.get('first_vs_ref_mae') is not None and float(summary['first_vs_ref_mae']) > 1e-4:
        summary['warnings'].append('cache_sigma0_not_reference')
    summary['level'] = _rf_diagnostic_level(summary)
    return summary


def _rf_print_stability_summary(summary: Dict[str, Any]) -> None:
    level = summary.get('level', 'WARN')
    print(f'{_PREFIX}   RF numerical sanity: {level}')
    print(
        f'{_PREFIX}     cache_items={summary.get("cache_items")}  '
        f'sigmas={summary.get("sigmas")}  '
        f'first_sigma={_rf_scalar_fmt(summary.get("first_sigma"))}  '
        f'last_sigma={_rf_scalar_fmt(summary.get("last_sigma"))}  '
        f'finite_all={summary.get("finite_all")}'
    )
    print(
        f'{_PREFIX}     std: ref={_rf_scalar_fmt(summary.get("ref_std"))}  '
        f'first={_rf_scalar_fmt(summary.get("first_std"))}  '
        f'final={_rf_scalar_fmt(summary.get("final_std"))}  '
        f'eps={_rf_scalar_fmt(summary.get("eps_std"))}  '
        f'final/eps={_rf_scalar_fmt(summary.get("final_eps_std_ratio"))}  '
        f'range=[{_rf_scalar_fmt(summary.get("std_min"))}, {_rf_scalar_fmt(summary.get("std_max"))}]'
    )
    print(
        f'{_PREFIX}     final: mean={_rf_scalar_fmt(summary.get("final_mean"))}  '
        f'min={_rf_scalar_fmt(summary.get("final_min"))}  '
        f'max={_rf_scalar_fmt(summary.get("final_max"))}  '
        f'max_abs={_rf_scalar_fmt(summary.get("final_max_abs"))}  '
        f'max_abs_over_path={_rf_scalar_fmt(summary.get("max_abs_max"))}'
    )
    print(
        f'{_PREFIX}     deltas: mean|Δz|={_rf_scalar_fmt(summary.get("dz_mean"))}  '
        f'max|Δz|={_rf_scalar_fmt(summary.get("dz_max"))}  '
        f'first_vs_ref_mae={_rf_scalar_fmt(summary.get("first_vs_ref_mae"))}  '
        f'final_vs_eps_mae={_rf_scalar_fmt(summary.get("final_vs_eps_mae"))}  '
        f'final_vs_eps_rmse={_rf_scalar_fmt(summary.get("final_vs_eps_rmse"))}  '
        f'final_vs_eps_cos={_rf_scalar_fmt(summary.get("final_vs_eps_cosine"))}'
    )
    warnings = summary.get('warnings') or []
    if warnings:
        print(f'{_PREFIX}     warnings={warnings} bad_sigma_keys={summary.get("bad_sigma_keys", [])}')
    if level != 'PASS':
        print(
            f'{_PREFIX}   ⚠ RF numerical sanity did not PASS. This is a diagnostic flag only; '
            f'it means inspect the printed stats and the generated image before trusting the run.'
        )
    else:
        print(
            f'{_PREFIX}   RF numerical sanity PASS only means no obvious numeric collapse/divergence; '
            f'it does not prove visual/semantic quality.'
        )


def _rf_model_identity(model_patcher: Any) -> Dict[str, Any]:
    """Best-effort model identity diagnostics; never used for math decisions."""
    base = getattr(model_patcher, 'model', model_patcher)
    diffusion_model = getattr(base, 'diffusion_model', None)
    model_config = getattr(base, 'model_config', None)
    unet_config = getattr(model_config, 'unet_config', None)
    if not isinstance(unet_config, dict):
        unet_config = {}
    model_type = getattr(base, 'model_type', None)
    model_sampling = getattr(base, 'model_sampling', None)
    latent_format = getattr(base, 'latent_format', None)
    info = {
        'base_class': type(base).__name__ if base is not None else 'None',
        'diffusion_class': type(diffusion_model).__name__ if diffusion_model is not None else 'None',
        'diffusion_module': getattr(type(diffusion_model), '__module__', '') if diffusion_model is not None else '',
        'model_type': getattr(model_type, 'name', str(model_type)),
        'model_sampling_class': type(model_sampling).__name__ if model_sampling is not None else 'None',
        'latent_format_class': type(latent_format).__name__ if latent_format is not None else 'None',
        'image_model': unet_config.get('image_model', None),
        'in_channels': unet_config.get('in_channels', None),
        'out_channels': unet_config.get('out_channels', None),
    }
    return info


def _rf_print_model_identity(prefix: str, info: Dict[str, Any]) -> None:
    print(
        f'{prefix} model_info: base={info.get("base_class")} '
        f'diffusion={info.get("diffusion_module")}.{info.get("diffusion_class")} '
        f'image_model={info.get("image_model")} adapter={info.get("architecture_name", info.get("architecture", "unknown"))}\n'
        f'{prefix} model_type={info.get("model_type")} '
        f'sampling={info.get("model_sampling_class")} '
        f'latent_format={info.get("latent_format_class")} '
        f'in_channels={info.get("in_channels")} out_channels={info.get("out_channels")}'
    )


def _rf_step_iterator(num_steps: int):
    """Plain RF step iterator.

    Do not use tqdm/model_trange here because that refreshes a single terminal
    line. RF inversion wants persistent per-step console lines, while still
    keeping the preview callback independent.
    """
    return range(max(0, int(num_steps)))

def _rf_format_duration(seconds: float) -> str:
    seconds_i = int(max(0, round(float(seconds))))
    minutes, seconds_i = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f'{hours:d}:{minutes:02d}:{seconds_i:02d}'
    return f'{minutes:02d}:{seconds_i:02d}'

def _rf_progress_snapshot(step_i: int, total_steps: int, start_time: float, persistent: bool = False) -> None:
    total_steps = max(1, int(total_steps))
    step_i = max(0, min(int(step_i), total_steps))

    elapsed = max(0.0, time.time() - float(start_time))
    frac = step_i / total_steps

    bar_width = 70
    filled = int(round(bar_width * frac))
    bar = '█' * filled + ' ' * (bar_width - filled)

    percent = int(round(frac * 100.0))
    rate = step_i / elapsed if elapsed > 1e-9 else 0.0
    remaining = max(0.0, (total_steps - step_i) / rate) if step_i > 0 and rate > 1e-9 else 0.0
    rate_text = f'{rate:.2f}it/s' if rate >= 1.0 else f'{(1.0 / max(rate, 1e-9)):.2f}s/it'

    line = (
        f'RF inversion: {percent:3d}%|{bar}| '
        f'{step_i}/{total_steps} '
        f'[{_rf_format_duration(elapsed)}<{_rf_format_duration(remaining)}, {rate_text}]'
    )

    end = '\n' if persistent or step_i >= total_steps else '\r'
    print(line, end=end, flush=True)
    
# ═══════════════════════════════════════════════════════════════════════════════
# Parameterization auto-detection
# ═══════════════════════════════════════════════════════════════════════════════

def _velocity_from_pred(
    x_sigma: torch.Tensor,
    pred: torch.Tensor,
    sigma: float,
    parameterization: str,
) -> torch.Tensor:
    """
    Convert ComfyUI ``model.apply_model`` output into the RF velocity dx/dsigma.

    ComfyUI's model_function_wrapper receives ``model.apply_model``. In current
    ComfyUI, BaseModel._apply_model returns ``model_sampling.calculate_denoised``;
    for supported rectified-flow models this is a denoised/x0-style tensor, not the raw transformer
    velocity. Therefore RF inversion must recover velocity from x_sigma and x0.

    Only the explicit opt-in label ``raw_velocity`` is treated as already being
    a velocity. RFInversion itself does not set that label.
    """
    mode = str(parameterization or 'x0').lower()
    if mode in ('raw_velocity', 'velocity_raw', 'model_velocity'):
        return pred

    sigma_f = max(float(sigma), 1e-7)
    return (x_sigma - pred) / sigma_f

# ═══════════════════════════════════════════════════════════════════════════════
# RF utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

_GAMMA_RF_MODES = {'rf_gamma', 'rf_gamma_rk2'}

def _coerce_gamma_curve(value: Any = 0.0) -> float:
    """Clamp gamma_curve to the supported range."""
    try:
        curve = float(value)
    except Exception:
        curve = 0.0
    if not math.isfinite(curve):
        curve = 0.0
    return max(0.0, min(8.0, curve))

def _normalize_rf_mode_and_gamma_curve(
    mode: str,
    gamma_curve: float = 0.0,
) -> Tuple[str, float]:
    """Normalize the RF mode string and clamp gamma_curve."""
    mode = str(mode or 'rf_gamma')
    return mode, _coerce_gamma_curve(gamma_curve)

def _coerce_norm_strength(norm_strength: float) -> float:
    try:
        strength = float(norm_strength)
    except Exception:
        strength = 0.0
    if not math.isfinite(strength):
        strength = 0.0
    return max(0.0, min(1.0, strength))

def _rf_gamma_for_mode(
    mode: str,
    gamma: float,
    sigma_prev: float,
    sigma_cur: float,
    gamma_curve: float = 0.0,
) -> float:
    mode, gamma_curve = _normalize_rf_mode_and_gamma_curve(mode, gamma_curve)
    if mode in ('linear', 'fireflow'):
        return 0.0
    if gamma_curve > 0.0 and mode in _GAMMA_RF_MODES:
        s = max(0.0, min(1.0, 0.5 * (float(sigma_prev) + float(sigma_cur))))
        bell = max(0.0, min(1.0, 4.0 * s * (1.0 - s)))
        return float(gamma) * (bell ** gamma_curve)
    return float(gamma)

def _rf_linear_target(ref_clean: torch.Tensor, eps: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = max(0.0, min(1.0, float(sigma)))
    return (1.0 - sigma) * ref_clean + sigma * eps

def _rf_match_mean_std(x: torch.Tensor, target: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Blend x toward target's per-sample mean/std. Prevents RF feature drift."""
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return x
    dims = tuple(range(1, x.ndim))
    x_mean = x.mean(dim=dims, keepdim=True)
    x_std = x.std(dim=dims, keepdim=True).clamp_min(1e-6)
    t_mean = target.mean(dim=dims, keepdim=True)
    t_std = target.std(dim=dims, keepdim=True).clamp_min(1e-6)
    matched = (x - x_mean) / x_std * t_std + t_mean
    return (1.0 - strength) * x + strength * matched

# ═══════════════════════════════════════════════════════════════════════════════
# PMI — Proximal-Mean Inversion (Wang et al., ICLR 2026)
# "Free Lunch for Stabilizing Rectified Flow Inversion"
# https://arxiv.org/abs/2602.11850
# ═══════════════════════════════════════════════════════════════════════════════

class _PMIState:
    """Carries running velocity mean across steps for PMI inversion."""
    def __init__(self) -> None:
        self.v_mean: Optional[torch.Tensor] = None
        self.step_count: int = 0
        self.v_norm_sq_mean: float = 0.0

    def reset(self) -> None:
        self.v_mean = None
        self.step_count = 0

    def update_and_correct(
        self,
        v_model: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Update the running mean and return the PMI-corrected velocity.

        alpha: blend weight toward the running mean (0 = pure model, 1 = pure mean).
               Paper suggests ~0.3–0.5 gives best stability without loss of fidelity.
        """
        alpha = max(0.0, min(1.0, float(alpha)))

        k = self.step_count  # steps seen so far, 0-indexed before update

        # ── Cumulative arithmetic mean (paper eq.) ───────────────────────
        if self.v_mean is None:
            self.v_mean = v_model.detach().clone()
            self.v_norm_sq_mean = float(v_model.detach().float().pow(2).mean().item())
            self.step_count = 1
            return v_model

        # incremental update: v̄_k = v̄_{k-1} * (k-1)/k + v_k / k
        k_new = k + 1
        self.v_mean = (self.v_mean * (k / k_new)
                    + v_model.detach() * (1.0 / k_new)).to(
                        device=v_model.device, dtype=v_model.dtype)
        self.v_norm_sq_mean = (
            self.v_norm_sq_mean * (k / k_new)
            + float(v_model.detach().float().pow(2).mean().item()) * (1.0 / k_new)
        )
        self.step_count = k_new

        # ── Linear blend toward mean ─────────────────────────────────────
        v_corrected = (1.0 - alpha) * v_model + alpha * self.v_mean

        # ── Spherical Gaussian projection (paper constraint) ─────────────
        # The paper keeps v_corrected within a ball of radius = ||v_model - v̄||
        # centred on v̄, so the blend never overshoots the model velocity.
        delta_model = v_model - self.v_mean
        delta_corr  = v_corrected - self.v_mean

        r_sq = float(delta_model.detach().float().pow(2).mean().item())
        c_sq = float(delta_corr.detach().float().pow(2).mean().item())

        if c_sq > r_sq and r_sq > 0.0:
            scale = math.sqrt(r_sq / c_sq)
            v_corrected = self.v_mean + scale * delta_corr

        return v_corrected
    
# ═══════════════════════════════════════════════════════════════════════════════
# Main RF trajectory builder
# ═══════════════════════════════════════════════════════════════════════════════

def _rf_build_cache_from_sampler_sigmas(
    ref_clean:      torch.Tensor,
    sampler_sigmas: List[float],
    apply_model_fn: Callable,
    base_model_kwargs: Dict[str, Any],
    gamma:          float = 0.5,
    seed:           int   = 0,
    stats:          Optional[_RuntimeStats] = None,
    eps:            Optional[torch.Tensor] = None,
    rf_mode:        str   = 'rf_gamma',
    norm_strength:  float = 0.0,
    pmi_alpha:      float = 0.4,
    gamma_curve:     float = 0.0,
    preview_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]] = None,
) -> Tuple[Dict[float, torch.Tensor], torch.Tensor, List[float]]:
    """
    Build reference x_sigma latents on the actual sampler sigma grid.
    """
    norm_strength = _coerce_norm_strength(norm_strength)
    mode, gamma_curve = _normalize_rf_mode_and_gamma_curve(rf_mode, gamma_curve)
    valid_modes = {'linear', 'rf_gamma', 'rf_gamma_rk2', 'fireflow'}
    if mode not in valid_modes:
        print(f'{_PREFIX}   ⚠ Unknown rf_mode={mode!r}; falling back to rf_gamma')
        mode = 'rf_gamma'

    parameterization = getattr(stats, 'parameterization', 'unknown') if stats else 'unknown'

    device = ref_clean.device
    dtype  = ref_clean.dtype

    if eps is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        eps = torch.randn(ref_clean.shape, device=device, dtype=dtype, generator=rng)

    # Build sorted unique sigma grid starting from 0.
    sigmas: List[float] = [0.0]
    for s in sampler_sigmas:
        try:
            sf = max(0.0, min(1.0, float(s)))
        except Exception:
            continue
        if all(abs(sf - existing) > 1e-6 for existing in sigmas):
            sigmas.append(sf)
    sigmas = sorted(sigmas)

    z = ref_clean.clone()
    prev = 0.0
    cache: Dict[float, torch.Tensor] = {0.0: z.detach().clone()}
    model_ok = 0
    failures = 0
    vm_sum = 0.0
    vp_sum = 0.0
    # FireFlow state: stores midpoint velocity from previous step for reuse.
    next_step_velocity: Optional[torch.Tensor] = None

    # PMI state. pmi_alpha is the on/off control: <= 0 means disabled.
    pmi_alpha_eff = max(0.0, min(1.0, float(pmi_alpha)))
    use_pmi = pmi_alpha_eff > 0.0
    pmi_state = _PMIState()
    total_preview_steps = max(1, len(sigmas) - 1)
    previewed_steps: set = set()

    def _preview_once(step_index: int, raw_pred: Optional[torch.Tensor], x_current: Optional[torch.Tensor]) -> None:
        if preview_callback is None or raw_pred is None:
            return
        step_index = max(0, min(total_preview_steps - 1, int(step_index)))
        if step_index in previewed_steps:
            return
        previewed_steps.add(step_index)
        _rf_emit_preview(preview_callback, step_index, raw_pred, x_current, total_preview_steps)

    _rf_vprint(stats,
        f'{_PREFIX}   RF trajectory mode: {mode}  gamma={gamma:.4f}  '
        f'gamma_curve={gamma_curve:.3f}  '
        f'norm_strength={norm_strength:.3f}  '
        f'norm={"on" if norm_strength > 0.0 else "off"}  '
        f'parameterization={parameterization}\n'
        f'{_PREFIX}   pmi_alpha={pmi_alpha_eff:.3f}  '
        f'PMI={"on" if use_pmi else "off"}'
    )

    # Print persistent RF inversion progress snapshots. This keeps every RF step
    rf_total_steps = max(1, len(sigmas) - 1)
    rf_progress_start_time = time.time()

    for step_index in _rf_step_iterator(rf_total_steps):
        step_i = int(step_index) + 1
        s = sigmas[step_i]
        sigma_prev = float(prev)
        sigma_cur  = float(s)
        delta      = float(sigma_cur - sigma_prev)
        z_prev     = z.detach().clone()
        gamma_eff  = _rf_gamma_for_mode(mode, gamma, sigma_prev, sigma_cur, gamma_curve)
    
        vm_abs = 0.0
        vp_abs = 0.0
        extra  = ''
    
        # ── Helper: run model and convert output to velocity ─────────────────
        def _call_model_as_velocity(z_in, sigma_val, label=''):
            nonlocal model_ok, failures, vm_sum
            t_tensor = torch.full((z_in.shape[0],), sigma_val, device=device, dtype=dtype)
            with torch.no_grad():
                try:
                    raw = apply_model_fn(z_in, t_tensor, **base_model_kwargs)
                    model_ok += 1
                    v = _velocity_from_pred(z_in, raw, sigma_val, parameterization)
                    vm_sum += float(v.abs().mean().item())
                    return v, True, raw
                except Exception as exc:
                    failures += 1
                    print(f'{_PREFIX}   [WARNING {mode}{label}] model failed at σ={sigma_val:.6f}: {exc}')
                    return torch.zeros_like(z_in), False, None
    
        def _apply_pmi_if_enabled(v: torch.Tensor) -> torch.Tensor:
            if not use_pmi:
                return v
            return pmi_state.update_and_correct(v, alpha=pmi_alpha_eff)
    
        if mode == 'linear':
            z = _rf_linear_target(ref_clean, eps, sigma_cur)
            extra = 'linear_target'
    
        elif mode == 'fireflow':
            # ── (Deng et al., ICML 2025) ─────────
            if next_step_velocity is None:
                v_pred, ok, raw_preview = _call_model_as_velocity(z, sigma_prev, ' fresh')
                vm_abs = float(v_pred.abs().mean().item())
                pred_source = 'fresh'
            else:
                v_pred = next_step_velocity.to(device=device, dtype=dtype)
                vm_abs = float(v_pred.abs().mean().item())
                pred_source = 'reused_mid'
    
            z_mid      = z + 0.5 * delta * v_pred
            sigma_mid  = sigma_prev + 0.5 * delta
            v_mid, ok, raw_preview_mid = _call_model_as_velocity(z_mid, sigma_mid, ' mid')
            vm_abs_mid = float(v_mid.abs().mean().item())
    
            v_mid_total = _apply_pmi_if_enabled(v_mid)
            next_step_velocity = v_mid_total.detach().clone()
            z = z + delta * v_mid_total
            _preview_once(step_i - 1, raw_preview_mid, z)
            extra = (
                f'FireFlow pred={pred_source}  σ_mid={sigma_mid:.6f}  '
                f'|v_pred|={vm_abs:.5f}  |v_mid|={vm_abs_mid:.5f}'
            )
            if use_pmi:
                extra += f'  PMI step={pmi_state.step_count}'
    
        else:
            # ── RF-style velocity  ─
            v_model, ok, raw_preview = _call_model_as_velocity(z, sigma_prev)
            vm_abs = float(v_model.abs().mean().item())
    
            denom   = max(1.0 - sigma_prev, 1e-7)
            v_prior = (eps - z) / denom
            vp_abs  = float(v_prior.abs().mean().item())
            vp_sum += vp_abs
    
            if mode == 'rf_gamma_rk2':
                v1    = gamma_eff * v_model + (1.0 - gamma_eff) * v_prior
                z_mid = z + 0.5 * delta * v1
                sigma_mid = sigma_prev + 0.5 * delta
                v_model_mid, ok_mid, raw_preview_mid = _call_model_as_velocity(z_mid, sigma_mid, ' mid')
                vm_abs_mid = float(v_model_mid.abs().mean().item())
    
                denom_mid = max(1.0 - sigma_mid, 1e-7)
                v_prior_mid = (eps - z_mid) / denom_mid
                vp_abs_mid  = float(v_prior_mid.abs().mean().item())
                vp_sum += vp_abs_mid
    
                v_total = gamma_eff * v_model_mid + (1.0 - gamma_eff) * v_prior_mid
                v_total = _apply_pmi_if_enabled(v_total)
                z = z + delta * v_total
                _preview_once(step_i - 1, raw_preview_mid, z)
                extra = f'mid |v_model_mid|={vm_abs_mid:.5f}'
                if use_pmi:
                    extra += f'  PMI step={pmi_state.step_count}'
            else:
                v_total = gamma_eff * v_model + (1.0 - gamma_eff) * v_prior
                v_total = _apply_pmi_if_enabled(v_total)
                z = z + delta * v_total
                _preview_once(step_i - 1, raw_preview, z)
                if use_pmi:
                    extra = f'PMI step={pmi_state.step_count}'
    
        if norm_strength > 0.0:
            target = _rf_linear_target(ref_clean, eps, sigma_cur)
            z = _rf_match_mean_std(z, target, strength=norm_strength)
            extra = (extra + '  ' if extra else '') + f'norm={norm_strength:.2f}'
    
        prev  = sigma_cur
        z_mean = float(z.mean().item())
        z_std  = float(z.std().item())
        z_min  = float(z.min().item())
        z_max  = float(z.max().item())
        dz_abs = float((z - z_prev).abs().mean().item())
        cache[round(sigma_cur, 6)] = z.detach().clone()
    
        _rf_vprint(stats,
            f'{_PREFIX}     z_sigma step {step_i:02d}/{len(sigmas)-1}: '
            f'mode={mode}  γ_eff={gamma_eff:.4f}  '
            f'σ_prev={sigma_prev:.6f} -> σ={sigma_cur:.6f}  Δσ={delta:.6f}  '
            f'|model|={vm_abs:.5f}  |prior|={vp_abs:.5f}  |Δz|={dz_abs:.5f}  {extra}\n'
            f'{_PREFIX}       z_σ mean={z_mean:.4f}  std={z_std:.4f}  '
            f'min={z_min:.4f}  max={z_max:.4f}'
        )

        _rf_progress_snapshot(
            step_i,
            rf_total_steps,
            rf_progress_start_time,
            persistent=_coerce_bool(getattr(stats, 'rf_verbose', False)),
        )
    
    steps = max(1, len(sigmas) - 1)
    _rf_vprint(stats,
        f'{_PREFIX}   RF schedule build: mode={mode}  sampler_sigmas={len(sampler_sigmas)}  '
        f'unique={len(sigmas)}  rf_steps={len(sigmas)-1}  '
        f'model_ok={model_ok}  failures={failures}\n'
        f'{_PREFIX}     sigma_range=[{sigmas[0]:.6f}, {sigmas[-1]:.6f}]  '
        f'|model|={vm_sum/max(1, model_ok):.5f}  |prior|={vp_sum/steps:.5f}  '
        f'z_final std={z.std().item():.4f}  parameterization={parameterization}'
    )
    if failures > 0:
        print(f'{_PREFIX}   ⚠ RF schedule warning: {failures} model call(s) failed.')

    return cache, eps, sigmas

def _rf_increment_reference_one_step(
    z_prev:         torch.Tensor,
    sigma_prev:     float,
    sigma_cur:      float,
    apply_model_fn: Callable,
    base_model_kwargs: Dict[str, Any],
    gamma:          float = 0.5,
    seed:           int   = 0,
    stats:          Optional[_RuntimeStats] = None,
    eps:            Optional[torch.Tensor] = None,
    rf_mode:        str   = 'rf_gamma',
    gamma_curve: float = 0.0,
    norm_strength: float = 0.0,
    preview_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stable direct one-step RF reference latent for the CURRENT sampler sigma.
    Used as a fallback when the sampler wrapper didn't capture the full sigma schedule.
    """
    sigma_cur = max(0.0, min(1.0, float(sigma_cur)))
    delta_sigma = sigma_cur

    device = z_prev.device
    dtype  = z_prev.dtype
    parameterization = getattr(stats, 'parameterization', 'unknown') if stats else 'unknown'

    if eps is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        eps = torch.randn(z_prev.shape, device=device, dtype=dtype, generator=rng)

    t_tensor = torch.zeros((z_prev.shape[0],), device=device, dtype=dtype)
    raw = None
    with torch.no_grad():
        try:
            raw     = apply_model_fn(z_prev, t_tensor, **base_model_kwargs)
            v_model = _velocity_from_pred(z_prev, raw, 0.0, parameterization)
            v_model_abs = float(v_model.abs().mean().item())
            model_ok = True
        except Exception as exc:
            print(f'{_PREFIX}   [WARNING direct RF] model call failed at σ=0.000000: {exc}')
            v_model = torch.zeros_like(z_prev)
            v_model_abs = 0.0
            model_ok = False

    v_prior = eps - z_prev
    v_prior_abs = float(v_prior.abs().mean().item())
    gamma_eff = _rf_gamma_for_mode(rf_mode, gamma, sigma_prev, sigma_cur, gamma_curve)
    v_total = gamma_eff * v_model + (1.0 - gamma_eff) * v_prior
    z_cur   = z_prev + delta_sigma * v_total
    norm_strength = max(0.0, min(1.0, float(norm_strength)))
    if norm_strength > 0.0:
        target = _rf_linear_target(z_prev, eps, sigma_cur)
        z_cur = _rf_match_mean_std(z_cur, target, strength=norm_strength)

    _rf_emit_preview(preview_callback, 0, raw, z_cur, 1)

    _rf_vprint(stats,
        f'{_PREFIX}   RF direct σ_base=0.000000 -> σ={sigma_cur:.6f}  '
        f'Δσ={delta_sigma:.6f}  gamma_eff={gamma_eff:.4f}  '
        f'norm_strength={norm_strength:.3f}  '
        f'model_ok={model_ok}  parameterization={parameterization}\n'
        f'{_PREFIX}     |v_model|={v_model_abs:.5f}  |v_prior|={v_prior_abs:.5f}  '
        f'z_cur std={z_cur.std().item():.4f}'
    )
    return z_cur, eps

def _normalize_sigma_float(value: Any) -> Optional[float]:
    try:
        if torch.is_tensor(value):
            v = float(value.detach().float().mean().item())
        else:
            v = float(value)
        if not math.isfinite(v):
            return None
        if 0.0 <= v <= 1.0:
            return max(0.0, min(1.0, v))
        if 1.0 < v <= 1000.0:
            return max(0.0, min(1.0, v / 1000.0))
    except Exception:
        return None
    return None

def _coerce_sigma_sequence(value: Any) -> Optional[List[float]]:
    """Convert a scheduler sigma/timestep list into normalized [0,1] floats."""
    try:
        if value is None:
            return None
        if torch.is_tensor(value):
            flat = value.detach().float().flatten().tolist()
        elif isinstance(value, (list, tuple)):
            flat = []
            for item in value:
                if torch.is_tensor(item):
                    flat.extend(item.detach().float().flatten().tolist())
                elif isinstance(item, (int, float)):
                    flat.append(float(item))
                else:
                    return None
        else:
            return None
        out: List[float] = []
        for item in flat:
            s = _normalize_sigma_float(item)
            if s is not None:
                out.append(s)
        if len(out) < 2:
            return None
        dedup: List[float] = []
        for s in out:
            if not dedup or abs(dedup[-1] - s) > 1e-6:
                dedup.append(s)
        return dedup if len(dedup) >= 2 else None
    except Exception:
        return None

def _find_sigma_schedule(obj: Any, depth: int = 0) -> Optional[List[float]]:
    if depth > 6 or obj is None:
        return None

    if isinstance(obj, dict):
        preferred = (
            'sample_sigmas', 'sampler_sigmas', 'sigmas', 'scheduler_sigmas',
            'denoise_sigmas', 'noise_sigmas', 'timesteps', 'timestep_schedule',
        )
        for key in preferred:
            if key in obj:
                seq = _coerce_sigma_sequence(obj.get(key))
                if seq is not None:
                    return seq
        for key, value in obj.items():
            key_l = str(key).lower()
            if any(word in key_l for word in ('sigma', 'timestep', 'schedule')):
                seq = _coerce_sigma_sequence(value)
                if seq is not None:
                    return seq
            if isinstance(value, dict):
                found = _find_sigma_schedule(value, depth + 1)
                if found is not None:
                    return found
            elif isinstance(value, (list, tuple)) and any(
                word in key_l for word in ('sigma', 'timestep', 'schedule')
            ):
                found = _find_sigma_schedule(value, depth + 1)
                if found is not None:
                    return found

    if isinstance(obj, (list, tuple)):
        seq = _coerce_sigma_sequence(obj)
        if seq is not None:
            return seq
        for item in obj:
            if isinstance(item, dict):
                found = _find_sigma_schedule(item, depth + 1)
                if found is not None:
                    return found
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _select_model_adapter(model_patcher: Any, model_info: Optional[Dict[str, Any]] = None) -> Any:
    adapter = model_adapters.identify(model_patcher, model_info or {})
    if isinstance(model_info, dict):
        model_info['architecture'] = model_adapters.adapter_key(adapter)
        model_info['architecture_name'] = model_adapters.adapter_label(adapter)
    return adapter

def _safe_get_diffusion_model(model_patcher: Any, adapter: Any) -> Any:
    return adapter.find_diffusion_model(model_patcher)

def _repeat_to_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    if x.shape[0] == batch:
        return x
    if comfy is not None and hasattr(comfy.utils, 'repeat_to_batch_size'):
        return comfy.utils.repeat_to_batch_size(x, batch)
    reps = math.ceil(batch / x.shape[0])
    return x.repeat((reps,) + (1,) * (x.ndim - 1))[:batch]

def _clone_model_options(options: Dict[str, Any]) -> Dict[str, Any]:
    out = options.copy()
    out['transformer_options'] = options.get('transformer_options', {}).copy()
    return out

def _clone_conditioning_for_rf(c: Dict[str, Any]) -> Dict[str, Any]:
    out = c.copy()
    to  = out.get('transformer_options', {})
    if isinstance(to, dict):
        to = to.copy()
        to.pop(_TRANSFORMER_CONFIG_KEY, None)
        out['transformer_options'] = to
    else:
        out['transformer_options'] = {}
    return out

def _slice_conditioning_batch(obj: Any, start: int, end: int) -> Any:
    if torch.is_tensor(obj):
        try:
            if obj.ndim > 0 and int(obj.shape[0]) >= end:
                return obj[start:end]
        except Exception:
            pass
        return obj
    if isinstance(obj, dict):
        return {
            k: (v if k == 'transformer_options' else _slice_conditioning_batch(v, start, end))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_slice_conditioning_batch(v, start, end) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_slice_conditioning_batch(v, start, end) for v in obj)
    return obj

def _build_rf_conditioning_kwargs(
    c: Dict[str, Any],
    ref_conditioning: Any,
    target_b: int,
) -> Tuple[Dict[str, Any], str]:
    try:
        if ref_conditioning is not None and target_b > 0:
            merged, _forced = _merge_reference_conditioning_into_c(c, ref_conditioning, target_b)
            ref_only = _slice_conditioning_batch(merged, target_b, target_b * 2)
            return _clone_conditioning_for_rf(ref_only), 'reference'
    except Exception as exc:
        print(f'{_PREFIX}   ⚠ RF conditioning fallback: {exc}')
    return _clone_conditioning_for_rf(c), 'target-fallback'

def _sigma_from_timestep(timestep: torch.Tensor) -> float:
    try:
        val = float(timestep.detach().float().mean().item())
        if math.isfinite(val):
            if 0.0 <= val <= 1.0:
                return max(0.0, min(1.0, val))
            if 1.0 < val <= 1000.0:
                return max(0.0, min(1.0, val / 1000.0))
    except Exception:
        pass
    return 1.0

def _sigma_to_progress(timestep: torch.Tensor) -> float:
    return max(0.0, min(1.0, 1.0 - _sigma_from_timestep(timestep)))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)

def _repeat_conditioning_tree(obj: Any, src: int, tgt: int) -> Any:
    if torch.is_tensor(obj):
        try:
            if obj.ndim > 0 and int(obj.shape[0]) == src:
                return _repeat_to_batch(obj, tgt)
        except Exception:
            pass
        return obj
    if isinstance(obj, dict):
        return {
            k: v if k in ('transformer_options', 'ref_latents', 'ref_contexts')
            else _repeat_conditioning_tree(v, src, tgt)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_repeat_conditioning_tree(v, src, tgt) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_repeat_conditioning_tree(v, src, tgt) for v in obj)
    return obj

_TEXT_CONDITIONING_KEYS = {
    'c_crossattn', 'crossattn', 'context', 'cap_feats', 'cond',
    'encoder_hidden_states', 'txt', 'text', 'text_embeddings',
}
_POOLED_CONDITIONING_KEYS = {
    'pooled_output', 'clip_pooled', 'pooled', 'y', 'vector',
}
_MASK_CONDITIONING_KEYS = {
    'attention_mask', 'crossattn_mask', 'c_crossattn_mask',
    'cap_mask', 'cond_mask', 'mask',
}
_NUM_TOKEN_KEYS = {
    'num_tokens', 'tokens_num', 'n_tokens', 'cap_num_tokens',
}
_CONDITIONING_META_ALIASES = {
    'pooled_output': ('pooled_output', 'clip_pooled', 'pooled', 'y', 'vector'),
    'clip_pooled':   ('clip_pooled',   'pooled_output', 'pooled', 'y', 'vector'),
    'pooled':        ('pooled',        'pooled_output', 'clip_pooled', 'y', 'vector'),
    'y':             ('y',             'pooled_output', 'clip_pooled', 'pooled', 'vector'),
    'vector':        ('vector',        'pooled_output', 'clip_pooled', 'pooled', 'y'),
    'attention_mask':   ('attention_mask',   'crossattn_mask', 'c_crossattn_mask', 'cap_mask', 'mask'),
    'crossattn_mask':   ('crossattn_mask',   'attention_mask', 'c_crossattn_mask', 'cap_mask', 'mask'),
    'c_crossattn_mask': ('c_crossattn_mask', 'attention_mask', 'crossattn_mask',   'cap_mask', 'mask'),
    'cap_mask':         ('cap_mask',         'attention_mask', 'crossattn_mask',   'c_crossattn_mask', 'mask'),
    'mask':             ('mask',             'attention_mask', 'crossattn_mask',   'c_crossattn_mask', 'cap_mask'),
    'num_tokens':     ('num_tokens',     'tokens_num', 'n_tokens', 'cap_num_tokens'),
    'tokens_num':     ('tokens_num',     'num_tokens', 'n_tokens', 'cap_num_tokens'),
    'n_tokens':       ('n_tokens',       'num_tokens', 'tokens_num', 'cap_num_tokens'),
    'cap_num_tokens': ('cap_num_tokens', 'num_tokens', 'tokens_num', 'n_tokens'),
}

def _first_tensor_in_conditioning_entry(entry: Any) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if torch.is_tensor(entry):
        return entry, meta
    if isinstance(entry, dict):
        meta.update(entry)
        for key in ('c_crossattn', 'crossattn', 'conditioning', 'cond', 'context', 'cap_feats'):
            value = entry.get(key)
            if torch.is_tensor(value):
                return value, meta
        for value in entry.values():
            if torch.is_tensor(value) and value.ndim >= 2:
                return value, meta
        return None, meta
    if isinstance(entry, (list, tuple)):
        cond: Optional[torch.Tensor] = None
        for item in entry:
            if torch.is_tensor(item) and cond is None:
                cond = item
            elif isinstance(item, dict):
                meta.update(item)
        if cond is not None:
            return cond, meta
        for item in entry:
            cond, nested_meta = _first_tensor_in_conditioning_entry(item)
            if nested_meta:
                meta.update(nested_meta)
            if cond is not None:
                return cond, nested_meta
    return None, meta

def _extract_reference_conditioning(ref_conditioning: Any) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    if ref_conditioning is None:
        return None, {}
    if torch.is_tensor(ref_conditioning) or isinstance(ref_conditioning, dict):
        return _first_tensor_in_conditioning_entry(ref_conditioning)
    if isinstance(ref_conditioning, (list, tuple)):
        merged_meta: Dict[str, Any] = {}
        for entry in ref_conditioning:
            cond, meta = _first_tensor_in_conditioning_entry(entry)
            if meta:
                merged_meta.update(meta)
            if cond is not None:
                return cond, merged_meta
        return None, merged_meta
    return None, {}


def _meta_get(meta: Dict[str, Any], key: str) -> Any:
    for alias in _CONDITIONING_META_ALIASES.get(key, (key,)):
        if alias in meta:
            return meta[alias]
    return None

def _as_tensor_like(value: Any, like: torch.Tensor) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(
            device=like.device,
            dtype=like.dtype if value.is_floating_point() else value.dtype,
        )
    try:
        return torch.as_tensor(value, device=like.device)
    except Exception:
        return None

def _coerce_ref_tensor_like_target(ref_value, target_value, target_b):
    ref = ref_value.to(
        device=target_value.device,
        dtype=target_value.dtype if ref_value.is_floating_point() else ref_value.dtype,
    )
    if target_value.ndim >= 2 and ref.ndim == target_value.ndim - 1:
        ref = ref.unsqueeze(0)
    if ref.ndim > 0 and int(ref.shape[0]) != target_b:
        ref = _repeat_to_batch(ref, target_b)
    if target_value.ndim >= 3 and ref.ndim >= 3:
        if int(ref.shape[1]) != int(target_value.shape[1]):
            ref = _pad_or_truncate_tokens(ref, int(target_value.shape[1]))
    elif target_value.ndim >= 2 and ref.ndim >= 2:
        if int(ref.shape[1]) != int(target_value.shape[1]):
            ref = _pad_or_truncate_tokens(ref, int(target_value.shape[1]))
    return ref

def _conditioning_mask_from_source(source, batch, padded_tokens, device):
    if source is None:
        return None
    if torch.is_tensor(source):
        x = source.detach().to(device=device)
        if x.ndim == 0:
            return _num_tokens_to_valid_mask(x, batch, padded_tokens, device)
        if x.ndim == 1:
            if x.numel() == batch and not x.is_floating_point():
                return _num_tokens_to_valid_mask(x, batch, padded_tokens, device)
            if x.numel() == 1:
                return _num_tokens_to_valid_mask(x, batch, padded_tokens, device)
            x = x.view(1, -1)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        if int(x.shape[0]) != batch:
            x = _repeat_to_batch(x, batch)
        if int(x.shape[1]) != padded_tokens:
            x = _pad_or_truncate_tokens(x, padded_tokens)
        if x.is_floating_point() and torch.any(x < 0):
            return (x >= 0).to(torch.bool)
        return x.to(torch.bool)
    if isinstance(source, (list, tuple)):
        try:
            return _conditioning_mask_from_source(
                torch.as_tensor(source, device=device), batch, padded_tokens, device
            )
        except Exception:
            return None
    try:
        return _num_tokens_to_valid_mask(int(source), batch, padded_tokens, device)
    except Exception:
        return None

def _target_valid_mask_from_c(c, target_b, padded_tokens, device):
    for key in ('attention_mask', 'crossattn_mask', 'c_crossattn_mask', 'cap_mask', 'mask'):
        mask = _conditioning_mask_from_source(c.get(key), target_b, padded_tokens, device)
        if mask is not None:
            return mask
    for key in ('num_tokens', 'tokens_num', 'n_tokens', 'cap_num_tokens'):
        mask = _conditioning_mask_from_source(c.get(key), target_b, padded_tokens, device)
        if mask is not None:
            return mask
    return torch.ones((target_b, padded_tokens), device=device, dtype=torch.bool)

def _reference_valid_mask_from_conditioning(ref_cond, ref_meta, target_b, padded_tokens, device):
    for key in ('attention_mask', 'crossattn_mask', 'c_crossattn_mask', 'cap_mask', 'mask'):
        mask = _conditioning_mask_from_source(
            _meta_get(ref_meta, key), target_b, padded_tokens, device
        )
        if mask is not None:
            return mask
    for key in ('num_tokens', 'tokens_num', 'n_tokens', 'cap_num_tokens'):
        mask = _conditioning_mask_from_source(
            _meta_get(ref_meta, key), target_b, padded_tokens, device
        )
        if mask is not None:
            return mask
    real_tokens = int(ref_cond.shape[1]) if ref_cond.ndim >= 2 else padded_tokens
    return _num_tokens_to_valid_mask(real_tokens, target_b, padded_tokens, device)

def _conditioning_counts_from_mask(mask):
    m = mask.to(torch.bool)
    if m.ndim == 1:
        m = m.view(1, -1)
    return m.long().sum(dim=1)

def _concat_batch_conditioning_value(key, value, ref_cond, ref_meta, target_b, forced_cap_mask):
    if not torch.is_tensor(value):
        if key in _NUM_TOKEN_KEYS:
            try:
                target_counts = torch.as_tensor(
                    value, device=forced_cap_mask.device, dtype=torch.long,
                ).flatten()
                if target_counts.numel() == 1:
                    target_counts = target_counts.repeat(target_b)
                elif target_counts.numel() != target_b:
                    target_counts = _repeat_to_batch(
                        target_counts.view(-1, 1), target_b
                    ).flatten()
                ref_counts = _conditioning_counts_from_mask(
                    forced_cap_mask[target_b:target_b * 2]
                )
                return torch.cat([target_counts, ref_counts], dim=0)
            except Exception:
                return _repeat_conditioning_tree(value, target_b, target_b * 2)
        if key in _MASK_CONDITIONING_KEYS:
            return forced_cap_mask
        return _repeat_conditioning_tree(value, target_b, target_b * 2)

    try:
        if value.ndim == 0 or int(value.shape[0]) != target_b:
            return value
    except Exception:
        return value

    ref_value: Optional[torch.Tensor] = None

    if key in _TEXT_CONDITIONING_KEYS or (
        value.ndim >= 3 and ref_cond.ndim >= 3
        and int(value.shape[-1]) == int(ref_cond.shape[-1])
    ):
        ref_value = ref_cond
    elif key in _POOLED_CONDITIONING_KEYS:
        meta_value = _meta_get(ref_meta, key)
        if meta_value is not None:
            ref_value = _as_tensor_like(meta_value, value)
    elif key in _MASK_CONDITIONING_KEYS:
        ref_value = forced_cap_mask[target_b:target_b * 2].to(
            device=value.device,
            dtype=value.dtype if value.is_floating_point() else torch.bool,
        )
        if value.is_floating_point() and torch.any(value < 0):
            ref_value = _mask_to_additive(ref_value.to(torch.bool), dtype=value.dtype)
    elif key in _NUM_TOKEN_KEYS:
        ref_value = _conditioning_counts_from_mask(
            forced_cap_mask[target_b:target_b * 2]
        ).to(device=value.device, dtype=value.dtype)

    if ref_value is None:
        ref_value = value

    ref_value = _coerce_ref_tensor_like_target(ref_value, value, target_b)
    return torch.cat([value, ref_value], dim=0)

def _merge_reference_conditioning_into_c(c, ref_conditioning, target_b):
    ref_cond, ref_meta = _extract_reference_conditioning(ref_conditioning)
    if ref_cond is None:
        raise RuntimeError(
            'ref_conditioning must be connected and must contain a valid '
            'CONDITIONING tensor when reference_latent is connected.'
        )

    target_text = None
    for key in ('c_crossattn', 'crossattn', 'context', 'cap_feats', 'cond',
                'encoder_hidden_states'):
        value = c.get(key)
        if (torch.is_tensor(value) and value.ndim >= 3
                and int(value.shape[0]) == target_b):
            target_text = value
            break

    if target_text is None:
        for key, value in c.items():
            if (
                key != 'transformer_options'
                and torch.is_tensor(value)
                and value.ndim >= 3
                and int(value.shape[0]) == target_b
                and ref_cond.ndim >= 3
                and int(value.shape[-1]) == int(ref_cond.shape[-1])
            ):
                target_text = value
                break

    if target_text is None:
        raise RuntimeError(
            'Could not find the target text-conditioning tensor in model kwargs.'
        )

    if ref_cond.ndim == target_text.ndim - 1:
        ref_cond = ref_cond.unsqueeze(0)

    if ref_cond.ndim < 3 or int(ref_cond.shape[-1]) != int(target_text.shape[-1]):
        raise RuntimeError(
            f'ref_conditioning incompatible shape {tuple(ref_cond.shape)} '
            f'vs {tuple(target_text.shape)}.'
        )

    padded_tokens   = int(target_text.shape[1])
    device          = target_text.device
    target_mask     = _target_valid_mask_from_c(c, target_b, padded_tokens, device)
    ref_mask        = _reference_valid_mask_from_conditioning(
        ref_cond, ref_meta, target_b, padded_tokens, device
    )
    forced_cap_mask = torch.cat([target_mask, ref_mask], dim=0).to(torch.bool)

    out: Dict[str, Any] = {}
    for key, value in c.items():
        if key == 'transformer_options':
            out[key] = value
            continue
        out[key] = _concat_batch_conditioning_value(
            key, value, ref_cond, ref_meta, target_b, forced_cap_mask
        )
    return out, forced_cap_mask

# ═══════════════════════════════════════════════════════════════════════════════
# Token / mask helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_or_truncate_tokens(x: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if x.ndim < 2:
        return x
    cur = int(x.shape[1])
    if cur == target_tokens:
        return x
    if cur > target_tokens:
        return x[:, :target_tokens, ...]
    pad_shape    = list(x.shape)
    pad_shape[1] = target_tokens - cur
    pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

def _num_tokens_to_valid_mask(num_tokens, batch, padded_tokens, device):
    if torch.is_tensor(num_tokens):
        counts = num_tokens.detach().to(device=device).flatten().long()
        if counts.numel() == 1:
            counts = counts.repeat(batch)
        elif counts.numel() != batch:
            counts = _repeat_to_batch(counts.view(-1, 1), batch).flatten().long()
    elif isinstance(num_tokens, (list, tuple)):
        counts = torch.tensor(num_tokens, device=device, dtype=torch.long).flatten()
        if counts.numel() == 1:
            counts = counts.repeat(batch)
        elif counts.numel() != batch:
            counts = _repeat_to_batch(counts.view(-1, 1), batch).flatten().long()
    else:
        counts = torch.full(
            (batch,),
            int(num_tokens) if num_tokens is not None else padded_tokens,
            device=device, dtype=torch.long,
        )
    counts = counts.clamp(min=0, max=padded_tokens)
    ar = torch.arange(padded_tokens, device=device).view(1, padded_tokens)
    return ar < counts.view(batch, 1)

def _coerce_forced_cap_mask_for_feats(forced_cap_mask, cap_feats):
    mask = forced_cap_mask.to(device=cap_feats.device)
    if mask.ndim == 1:
        mask = mask.view(1, -1)
    if mask.ndim > 0 and int(mask.shape[0]) != int(cap_feats.shape[0]):
        mask = _repeat_to_batch(mask, int(cap_feats.shape[0]))
    if mask.ndim == 2 and int(mask.shape[1]) != int(cap_feats.shape[1]):
        mask = _pad_or_truncate_tokens(mask, int(cap_feats.shape[1]))
    return mask.to(torch.bool)

def _mask_to_additive(valid_mask, dtype=torch.float32):
    valid = valid_mask.to(torch.bool)
    out   = torch.zeros(valid.shape, device=valid.device, dtype=dtype)
    return out.masked_fill(~valid, -10000.0)

def _build_joint_additive_mask_from_cap_mask(
    cap_valid_mask, seq_len, text_range, device, dtype=torch.float32
):
    if not torch.is_tensor(cap_valid_mask) or cap_valid_mask.ndim < 2:
        return None
    if text_range is None:
        return None
    ts, te = int(text_range[0]), int(text_range[1])
    ts = max(0, min(ts, int(seq_len)))
    te = max(ts, min(te, int(seq_len)))
    if te <= ts:
        return None
    cap_valid_mask = cap_valid_mask.to(device=device).to(torch.bool)
    batch      = int(cap_valid_mask.shape[0])
    text_slots = te - ts
    text_valid = torch.zeros((batch, text_slots), device=device, dtype=torch.bool)
    copy_len   = min(text_slots, int(cap_valid_mask.shape[1]))
    if copy_len > 0:
        text_valid[:, :copy_len] = cap_valid_mask[:, :copy_len]
    full_valid = torch.ones((batch, int(seq_len)), device=device, dtype=torch.bool)
    full_valid[:, ts:te] = text_valid
    return _mask_to_additive(full_valid, dtype=dtype)

# ═══════════════════════════════════════════════════════════════════════════════
# RoPE frequency scale vector
# ═══════════════════════════════════════════════════════════════════════════════

def _build_frequency_scale_vector(
    head_dim, axes_dims, high_scale, low_scale, beta, device, dtype
):
    if not axes_dims or sum(int(x) for x in axes_dims) != head_dim:
        axes_dims = [head_dim]
    axes_dims = [int(x) for x in axes_dims]
    is_3axis  = len(axes_dims) == 3
    pieces: List[torch.Tensor] = []
    for axis_idx, axis_dim in enumerate(axes_dims):
        n_pairs = axis_dim // 2
        if n_pairs <= 0:
            pieces.append(torch.ones(axis_dim, device=device, dtype=dtype))
            continue
        if is_3axis and axis_idx == 0:
            pair_scales = torch.full(
                (n_pairs,), float(low_scale), device=device, dtype=torch.float32
            )
        else:
            d_tilde = (
                torch.zeros(1, device=device, dtype=torch.float32)
                if n_pairs == 1
                else torch.linspace(0.0, 1.0, n_pairs, device=device, dtype=torch.float32)
            )
            pair_scales = high_scale + (low_scale - high_scale) * d_tilde.pow(float(beta))
        pieces.append(pair_scales.to(dtype=dtype).repeat_interleave(2))
        if axis_dim % 2:
            pieces.append(torch.ones(1, device=device, dtype=dtype))
    out = torch.cat(pieces, dim=0)
    if out.numel() >= head_dim:
        return out[:head_dim]
    return torch.nn.functional.pad(out, (0, head_dim - out.numel()), value=1.0)

# ═══════════════════════════════════════════════════════════════════════════════
# AdaIN helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _adain(target, style, eps=1e-6):
    t_mean = target.mean(dim=1, keepdim=True)
    s_mean = style.mean(dim=1, keepdim=True)
    t_std  = target.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    s_std  = style.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    return (target - t_mean) / t_std * s_std + s_mean

def _cross_batch_adain_qk(xq, xk, cfg, target_bsz, strength, eps=1e-6):
    if target_bsz <= 0 or xq.shape[0] < target_bsz * 2:
        return xq, xk
    a = max(0.0, min(1.0, strength))
    if a <= 0.0:
        return xq, xk
    seqlen = xq.shape[1]
    for s, e in (cfg.get('target_qk_adain_ranges') or []):
        s, e = max(0, int(s)), min(int(e), seqlen)
        if e <= s:
            continue
        q_t, k_t = xq[:target_bsz, s:e], xk[:target_bsz, s:e]
        q_r, k_r = xq[target_bsz:target_bsz*2, s:e], xk[target_bsz:target_bsz*2, s:e]
        xq[:target_bsz, s:e] = q_t * (1 - a) + _adain(q_t, q_r, eps) * a
        xk[:target_bsz, s:e] = k_t * (1 - a) + _adain(k_t, k_r, eps) * a
    return xq, xk

def _repeat_kv_heads_if_needed(k, v, q_heads):
    kv = k.shape[2]
    if kv == q_heads:
        return k, v
    if q_heads % kv != 0:
        raise RuntimeError(f'Cannot expand KV heads: q={q_heads}, kv={kv}')
    n = q_heads // kv
    k = k.unsqueeze(3).repeat(1, 1, 1, n, 1).flatten(2, 3)
    v = v.unsqueeze(3).repeat(1, 1, 1, n, 1).flatten(2, 3)
    return k, v

# ═══════════════════════════════════════════════════════════════════════════════
# Architecture detection
# ═══════════════════════════════════════════════════════════════════════════════

_ACTIVE_MODEL_ADAPTER: Any = None

def _is_joint_attention(m):
    adapter = _ACTIVE_MODEL_ADAPTER
    fn = getattr(adapter, 'is_joint_attention', None)
    return bool(callable(fn) and fn(m))

def _is_main_layers_attention_name(name, min_layer=0, max_layer=29):
    adapter = _ACTIVE_MODEL_ADAPTER
    fn = getattr(adapter, 'is_attention_name', None)
    return bool(callable(fn) and fn(name, min_layer, max_layer))

# ═══════════════════════════════════════════════════════════════════════════════
# Context-refiner cap_mask patch
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_context_refiner_mask_modules(dm, stats):
    refiner = getattr(dm, 'context_refiner', None)
    if refiner is None:
        return 0, 0, 0
    try:
        modules = list(refiner)
    except Exception:
        modules = [refiner]
    matched = installed = restored = 0
    for idx, module in enumerate(modules):
        if not hasattr(module, 'forward') or not callable(getattr(module, 'forward', None)):
            continue
        matched += 1
        if hasattr(module, '_untwist_orig_context_refiner_forward'):
            module.forward = module._untwist_orig_context_refiner_forward
            restored += 1
        else:
            module._untwist_orig_context_refiner_forward = module.forward
        original_forward = module._untwist_orig_context_refiner_forward

        def make_forward(orig, layer_index):
            def patched_forward(self, *args, **kwargs):
                transformer_options = kwargs.get('transformer_options', None)
                if transformer_options is None and len(args) >= 4 and isinstance(args[3], dict):
                    transformer_options = args[3]
                cfg = (
                    transformer_options.get(_TRANSFORMER_CONFIG_KEY)
                    if isinstance(transformer_options, dict) else None
                )
                forced_cap_mask = (
                    cfg.get('forced_cap_mask', None) if isinstance(cfg, dict) else None
                )
                if torch.is_tensor(forced_cap_mask):
                    args_list = list(args)
                    cap_feats = (
                        args_list[0]
                        if len(args_list) >= 1 and torch.is_tensor(args_list[0])
                        else None
                    )
                    if cap_feats is not None:
                        replacement_mask = _coerce_forced_cap_mask_for_feats(
                            forced_cap_mask, cap_feats
                        )
                        substituted = False
                        
                        # Helper to ensure the replacement mask matches the expected attention shape
                        def _align_mask(orig_m):
                            if torch.is_tensor(orig_m):
                                try:
                                    return replacement_mask.view(orig_m.shape)
                                except Exception:
                                    pass
                            if replacement_mask.ndim == 2:
                                return replacement_mask.unsqueeze(1).unsqueeze(1)
                            return replacement_mask

                        if len(args_list) >= 2:
                            if args_list[1] is None or torch.is_tensor(args_list[1]):
                                args_list[1] = _align_mask(args_list[1])
                                substituted  = True
                        else:
                            for key in ('cap_mask', 'mask', 'x_mask'):
                                if key in kwargs and (
                                    kwargs[key] is None or torch.is_tensor(kwargs[key])
                                ):
                                    kwargs[key] = _align_mask(kwargs[key])
                                    substituted = True
                                    break
                        if substituted:
                            stats.context_refiner_calls += 1
                            return orig(*args_list, **kwargs)
                return orig(*args, **kwargs)
            return patched_forward

        module.forward = types.MethodType(make_forward(original_forward, idx), module)
        installed += 1

    _vprint(stats,
        f'{_PREFIX} Context-refiner mask patch: '
        f'matched={matched} installed={installed} restored={restored}')
    return matched, installed, restored

# ═══════════════════════════════════════════════════════════════════════════════
# patchify_and_embed patch
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_patchify_and_embed(dm, stats):
    if hasattr(dm, '_untwist_orig_patchify'):
        dm.patchify_and_embed = dm._untwist_orig_patchify
    else:
        dm._untwist_orig_patchify = dm.patchify_and_embed
    original = dm._untwist_orig_patchify

    def patched(self, x, cap_feats, cap_mask, t, num_tokens,
                ref_latents=[], ref_contexts=[], siglip_feats=[],
                transformer_options={}, *args, **kwargs):

        cfg_pre = (
            transformer_options.get(_TRANSFORMER_CONFIG_KEY)
            if isinstance(transformer_options, dict) else None
        )
        forced_cap_mask = (
            cfg_pre.get('forced_cap_mask', None) if isinstance(cfg_pre, dict) else None
        )

        if torch.is_tensor(forced_cap_mask):
            cap_mask = forced_cap_mask.to(device=cap_feats.device)
            if cap_mask.ndim == 1:
                cap_mask = cap_mask.view(1, -1)
            if cap_mask.ndim > 0 and int(cap_mask.shape[0]) != int(cap_feats.shape[0]):
                cap_mask = _repeat_to_batch(cap_mask, int(cap_feats.shape[0]))
            if cap_mask.ndim == 2 and int(cap_mask.shape[1]) != int(cap_feats.shape[1]):
                cap_mask = _pad_or_truncate_tokens(cap_mask, int(cap_feats.shape[1]))

        result = original(x, cap_feats, cap_mask, t, num_tokens, *args,
                          ref_latents=ref_latents, ref_contexts=ref_contexts,
                          siglip_feats=siglip_feats,
                          transformer_options=transformer_options, **kwargs)
        stats.patchify_calls += 1

        try:
            img, mask, img_size, cap_size, freqs_cis, timestep_zero_index = result
            cfg = transformer_options.get(_TRANSFORMER_CONFIG_KEY)
            if not cfg or not cfg.get('enabled'):
                return result

            cfg['axes_dims'] = list(getattr(self, 'axes_dims', []))
            cfg['head_dim']  = (int(getattr(self, 'dim', 0))
                                // max(1, int(getattr(self, 'n_heads', 1))))
            cfg['seq_len']   = int(img.shape[1])
            cfg['patch_size']= int(getattr(self, 'patch_size', 2))
            try:
                cfg['rope_theta'] = float(
                    getattr(getattr(self, 'rope_embedder', None), 'theta', 10000.0)
                )
            except Exception:
                cfg['rope_theta'] = 10000.0

            p = cfg['patch_size']
            target_range = target_text_range = None
            ref_ranges:      List[Tuple[int,int]] = []
            ref_real_ranges: List[Tuple[int,int]] = []

            if timestep_zero_index:
                target_range = tuple(int(v) for v in timestep_zero_index[0])
                if len(timestep_zero_index) > 1:
                    target_text_range = tuple(int(v) for v in timestep_zero_index[1])
            else:
                try:
                    cap0 = int(cap_size[0]) if isinstance(cap_size, (list, tuple)) else int(cap_size)
                except Exception:
                    cap0 = 0
                target_text_range = (0, cap0) if cap0 > 0 else None
                target_range = (max(0, cap0), int(img.shape[1]))

            real_range = target_range
            if target_range is not None:
                ts, te = int(target_range[0]), int(target_range[1])
                try:
                    real_tok   = (x.shape[-2] // p) * (x.shape[-1] // p)
                    real_range = (ts, min(ts + real_tok, te))
                except Exception:
                    real_range = target_range
                cfg['target_real_range'] = real_range
                ref_ranges.append((ts, te))
                ref_real_ranges.append(real_range)

            cfg.update({
                'ref_k_ranges':           ref_ranges,
                'ref_real_ranges':        ref_real_ranges,
                'target_range':           target_range,
                'target_text_range':      target_text_range,
                'target_qk_adain_ranges':
                    [cfg.get('target_real_range', target_range)] if target_range else [],
            })

            forced_mask_for_joint = cfg.get('forced_cap_mask', None)
            if torch.is_tensor(forced_mask_for_joint):
                joint_mask = _build_joint_additive_mask_from_cap_mask(
                    forced_mask_for_joint, int(img.shape[1]),
                    target_text_range, img.device, dtype=torch.float32,
                )
                if torch.is_tensor(joint_mask):
                    cfg['forced_joint_x_mask'] = joint_mask
                    if mask is None:
                        mask   = joint_mask
                        result = (img, mask, img_size, cap_size, freqs_cis, timestep_zero_index)

            transformer_options[_TRANSFORMER_CONFIG_KEY] = cfg
        except Exception:
            pass

        return result

    dm.patchify_and_embed = types.MethodType(patched, dm)
    _vprint(stats, f'{_PREFIX} patchify_and_embed patched.')

# ═══════════════════════════════════════════════════════════════════════════════
# Attention module patch
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_joint_attention_modules(dm, stats):
    matched = installed = restored = 0
    patched_names: List[str] = []

    for name, module in dm.named_modules():
        if not _is_main_layers_attention_name(name, 0, 29):
            continue
        if not _is_joint_attention(module):
            _vprint(stats, f'{_PREFIX} SKIP {name} ({type(module).__name__})')
            continue

        matched += 1
        patched_names.append(name)

        if hasattr(module, '_untwist_orig_forward'):
            module.forward = module._untwist_orig_forward
            restored += 1
        else:
            module._untwist_orig_forward = module.forward
        original_forward = module._untwist_orig_forward

        def make_forward(orig, module_name):
            def patched_forward(self, x, x_mask, freqs_cis, transformer_options={}):
                cfg = (
                    transformer_options.get(_TRANSFORMER_CONFIG_KEY)
                    if isinstance(transformer_options, dict) else None
                )
                if not cfg or not cfg.get('enabled'):
                    return orig(x, x_mask, freqs_cis,
                                transformer_options=transformer_options)

                block_idx = int(transformer_options.get('block_index', -1))
                if not (int(cfg['start_block']) <= block_idx <= int(cfg['end_block'])):
                    return orig(x, x_mask, freqs_cis,
                                transformer_options=transformer_options)

                ref_ranges  = cfg.get('ref_real_ranges') or cfg.get('ref_k_ranges') or []
                target_bsz  = int(cfg.get('cross_batch_target_batch', 0))
                if not ref_ranges or target_bsz <= 0:
                    return orig(x, x_mask, freqs_cis,
                                transformer_options=transformer_options)

                bsz, seqlen, _ = x.shape
                if bsz < target_bsz * 2:
                    return orig(x, x_mask, freqs_cis,
                                transformer_options=transformer_options)

                if x_mask is None and torch.is_tensor(cfg.get('forced_joint_x_mask', None)):
                    try:
                        fjm = cfg['forced_joint_x_mask'].to(device=x.device)
                        if int(fjm.shape[0]) != bsz:
                            fjm = _repeat_to_batch(fjm, bsz)
                        if int(fjm.shape[-1]) != seqlen:
                            cur = int(fjm.shape[-1])
                            if cur > seqlen:
                                fjm = fjm[..., :seqlen]
                            else:
                                pad = torch.zeros(
                                    (*fjm.shape[:-1], seqlen - cur),
                                    device=fjm.device, dtype=fjm.dtype,
                                )
                                fjm = torch.cat([fjm, pad], dim=-1)
                        x_mask = fjm
                    except Exception:
                        x_mask = None

                stats.attn_calls += 1

                xq, xk, xv = torch.split(
                    self.qkv(x),
                    [self.n_local_heads    * self.head_dim,
                     self.n_local_kv_heads * self.head_dim,
                     self.n_local_kv_heads * self.head_dim],
                    dim=-1,
                )
                xq = self.q_norm(xq.view(bsz, seqlen, self.n_local_heads,    self.head_dim))
                xk = self.k_norm(xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim))
                xv =             xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

                progress   = float(cfg.get('progress', 0.0))
                high_scale = _lerp(cfg['high_scale_start'], cfg['high_scale_end'], progress)
                low_scale  = _lerp(cfg['low_scale_start'],  cfg['low_scale_end'],  progress)
                beta       = float(cfg.get('beta', 2.0))

                if cfg.get('apply_adain') and float(cfg.get('adain_strength', 0)) > 0:
                    xq, xk = xq.clone(), xk.clone()
                    xq, xk = _cross_batch_adain_qk(
                        xq, xk, cfg, target_bsz, float(cfg['adain_strength'])
                    )

                xq, xk = apply_rope(xq, xk, freqs_cis)

                scale_vec = _build_frequency_scale_vector(
                    self.head_dim, cfg.get('axes_dims') or [],
                    high_scale, low_scale, beta,
                    xk.device, xk.dtype,
                ).view(1, 1, 1, self.head_dim)

                ref_k_pieces, ref_v_pieces = [], []
                for s, e in ref_ranges:
                    s, e = max(0, int(s)), min(int(e), seqlen)
                    if e <= s:
                        continue
                    ref_k_pieces.append(xk[target_bsz:target_bsz*2, s:e] * scale_vec)
                    ref_v_pieces.append(xv[target_bsz:target_bsz*2, s:e])

                if not ref_k_pieces:
                    return orig(x, x_mask, freqs_cis,
                                transformer_options=transformer_options)

                xq_t = xq[:target_bsz]
                xk_t = torch.cat([xk[:target_bsz]] + ref_k_pieces, dim=1)
                xv_t = torch.cat([xv[:target_bsz]] + ref_v_pieces, dim=1)
                xk_t, xv_t = _repeat_kv_heads_if_needed(xk_t, xv_t, self.n_local_heads)

                mask_t = None
                if x_mask is not None:
                    try:
                        mask_t  = x_mask[:target_bsz]
                        ref_len = sum(int(pc.shape[1]) for pc in ref_k_pieces)
                        if mask_t.ndim >= 2:
                            padding = torch.zeros(
                                (*mask_t.shape[:-1], ref_len),
                                device=mask_t.device, dtype=mask_t.dtype,
                            )
                            mask_t = torch.cat([mask_t, padding], dim=-1)
                    except Exception:
                        mask_t = None

                out_t = optimized_attention_masked(
                    xq_t.movedim(1,2), xk_t.movedim(1,2), xv_t.movedim(1,2),
                    self.n_local_heads, mask_t,
                    skip_reshape=True, transformer_options=transformer_options,
                )

                xq_r = xq[target_bsz:target_bsz*2]
                xk_r, xv_r = _repeat_kv_heads_if_needed(
                    xk[target_bsz:target_bsz*2],
                    xv[target_bsz:target_bsz*2],
                    self.n_local_heads,
                )
                mask_r = None
                try:
                    if x_mask is not None and int(x_mask.shape[0]) >= target_bsz * 2:
                        mask_r = x_mask[target_bsz:target_bsz*2]
                except Exception:
                    pass
                out_r = optimized_attention_masked(
                    xq_r.movedim(1,2), xk_r.movedim(1,2), xv_r.movedim(1,2),
                    self.n_local_heads, mask_r,
                    skip_reshape=True, transformer_options=transformer_options,
                )

                outs = [out_t, out_r]
                if bsz > target_bsz * 2:
                    xq_e = xq[target_bsz*2:]
                    xk_e, xv_e = _repeat_kv_heads_if_needed(
                        xk[target_bsz*2:], xv[target_bsz*2:], self.n_local_heads
                    )
                    outs.append(optimized_attention_masked(
                        xq_e.movedim(1,2), xk_e.movedim(1,2), xv_e.movedim(1,2),
                        self.n_local_heads, None,
                        skip_reshape=True, transformer_options=transformer_options,
                    ))

                return self.out(torch.cat(outs, dim=0))
            return patched_forward

        module.forward = types.MethodType(make_forward(original_forward, name), module)
        setattr(module, '_untwist_v652_active', True)
        installed += 1

    _vprint(stats,
        f'{_PREFIX} Attention patch: matched={matched} '
        f'installed={installed} restored={restored}')
    for n in patched_names:
        _vprint(stats, f'{_PREFIX}   - {n}')

    assert installed > 0, (
        f'{_PREFIX} FATAL: No layers.0..29 attention modules patched.'
    )
    return matched, installed, restored

# ═══════════════════════════════════════════════════════════════════════════════
# ComfyUI Nodes — split RF inversion from Untwisting RoPE
# ═══════════════════════════════════════════════════════════════════════════════

def _rf_new_debug_store() -> Dict[str, Any]:
    """Reset and return the module-level RF debug store used by RFInversion runtime."""
    debug_store: Dict[str, Any] = _RF_LAST_DEBUG_STORE
    debug_store.clear()
    debug_store.update({
        'cache': {},
        'xhat_cache': {},
        'pred_cache': {},
        'xhat_plus_cache': {},
        'sampler_sigmas': None,
        'built_sigmas': None,
        'run_count': 0,
        'persistent_cache_key': None,
        'persistent_cache_hit': False,
        'parameterization': 'unknown',
        'apply_model_output': 'comfy_denoised_x0',
        'model_info': {},
        'wrapper_calls': 0,
        'last_sigma': None,
        'last_cond_mode': None,
        'last_cache_lookup': None,
        'last_error': None,
    })
    return debug_store

def _rf_make_preview_callback(model_for_preview: Any, total_steps: int) -> Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]]:
    """Create a ComfyUI-style latent preview callback for RF raw predictions."""
    total_steps = max(1, int(total_steps))
    if latent_preview is not None:
        try:
            return latent_preview.prepare_callback(model_for_preview, total_steps)
        except Exception as exc:
            print(f'{_PREFIX} ⚠ RF preview callback disabled: {exc}')
    try:
        pbar = comfy.utils.ProgressBar(total_steps)
        def _progress_only(step: int, x0: torch.Tensor, x: torch.Tensor, steps: int) -> None:
            pbar.update_absolute(step + 1, steps)
        return _progress_only
    except Exception as exc:
        print(f'{_PREFIX} ⚠ RF progress callback disabled: {exc}')
        return None

def _rf_emit_preview(
    callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]],
    step: int,
    raw_pred: Optional[torch.Tensor],
    x_current: Optional[torch.Tensor],
    total_steps: int,
) -> None:
    """Emit one RF raw-pred preview frame without breaking sampling if preview decoding fails."""
    if callback is None or not torch.is_tensor(raw_pred):
        return
    try:
        preview_latent = raw_pred[:1].detach()
        current = x_current[:1].detach() if torch.is_tensor(x_current) else preview_latent
        callback(int(step), preview_latent, current, int(total_steps))
    except Exception as exc:
        print(f'{_PREFIX} ⚠ RF preview frame failed at step {int(step) + 1}: {exc}')

def _rf_latent_get_config(rf_inversion: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[torch.Tensor], Optional[Any], str]:
    """Read RFInversion's LATENT metadata without exposing a custom Comfy type."""
    if not isinstance(rf_inversion, dict):
        return False, {}, {}, None, None, 'not-connected'
    cfg = rf_inversion.get('untwist_rf_config', None)
    state = rf_inversion.get('untwist_rf_state', None)
    ref_clean = rf_inversion.get('untwist_ref_clean', None)
    ref_conditioning = rf_inversion.get('untwist_ref_conditioning', None)
    if not isinstance(cfg, dict):
        return False, {}, {}, None, None, 'missing-config'
    if not isinstance(state, dict):
        state = {}
        rf_inversion['untwist_rf_state'] = state
    if not torch.is_tensor(ref_clean):
        return False, cfg, state, None, ref_conditioning, 'missing-ref-clean'
    return True, cfg, state, ref_clean, ref_conditioning, 'RFInversion LATENT'


def _adapter_helpers() -> Dict[str, Any]:
    return {
        'prefix': _PREFIX,
        'config_key': _TRANSFORMER_CONFIG_KEY,
        'lerp': _lerp,
        'cross_batch_adain_qk': _cross_batch_adain_qk,
        'build_frequency_scale_vector': _build_frequency_scale_vector,
        'patch_context_refiner_mask_modules': _patch_context_refiner_mask_modules,
        'patch_patchify_and_embed': _patch_patchify_and_embed,
        'patch_joint_attention_modules': _patch_joint_attention_modules,
    }

def _prepare_reference_conditioning_for_adapter(
    adapter: Any,
    ref_conditioning: Any,
    dm: Any,
    device,
    dtype,
    stats: Optional[_RuntimeStats] = None,
    label: str = '',
) -> Tuple[Any, str]:
    fn = getattr(adapter, 'prepare_reference_conditioning', None)
    if not callable(fn):
        return ref_conditioning, 'not-applicable'
    return fn(ref_conditioning, dm, device, dtype, stats, label=label, helpers=_adapter_helpers())

def _append_conditioning_status(mode: str, status: str) -> str:
    if status and status != 'not-applicable':
        return f'{mode};{status}'
    return mode

class RFInversion:
    CATEGORY = 'model_patches/Untwisting RoPE'
    RETURN_TYPES = ('MODEL', 'LATENT')
    RETURN_NAMES = ('model', 'rf_inversion')
    FUNCTION = 'build'
    DESCRIPTION = (
        'Stores RF inversion settings/reference data in a normal LATENT and captures '
        'the sampler sigma schedule internally. No SIGMAS input is required.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'reference_latent': ('LATENT',),
                'rf_mode': (['linear', 'rf_gamma', 'rf_gamma_rk2', 'fireflow'], {
                    'default': 'rf_gamma',
                    'tooltip': (
                        'linear: no model velocity; pmi_alpha has no effect.\n'
                        'rf_gamma/rf_gamma_rk2: use gamma and optional gamma_curve.\n'
                        'fireflow: FireFlow recurrence with optional PMI/norm.'
                    ),
                }),
                'gamma': ('FLOAT', {'default': 0.1, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'gamma_curve': ('FLOAT', {'default': 2.0, 'min': 0.0, 'max': 8.0, 'step': 0.05}),
                'norm_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
                'pmi_alpha': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
                'verbose': ('BOOLEAN', {'default': False}),
            },
            'optional': {
                'ref_conditioning': ('CONDITIONING',),
            },
        }

    def build(
        self,
        model,
        reference_latent,
        rf_mode='fireflow',
        gamma=0.3,
        gamma_curve=0.0,
        norm_strength=0.0,
        pmi_alpha=0.4,
        verbose=False,
        ref_conditioning=None,
    ):
        rf_mode, gamma_curve = _normalize_rf_mode_and_gamma_curve(rf_mode, gamma_curve)
        norm_strength = _coerce_norm_strength(norm_strength)
        verbose_flag = _coerce_bool(verbose)

        if not isinstance(reference_latent, dict) or 'samples' not in reference_latent:
            raise RuntimeError("reference_latent must be a ComfyUI LATENT dict with 'samples'.")

        ref_clean = reference_latent['samples'].detach().clone()
        ref_clean = model.model.process_latent_in(ref_clean)

        # model_function_wrapper is passed ComfyUI model.apply_model, which returns
        # the denoised/x0-style prediction after model_sampling.calculate_denoised.
        # Keep the raw model type only as diagnostics; RF velocity conversion uses x0.
        model_info = _rf_model_identity(model)
        adapter = _select_model_adapter(model, model_info)
        detected_param = 'x0'
        dm_for_ref = None
        try:
            dm_for_ref = _safe_get_diffusion_model(model, adapter)
        except Exception as exc:
            if verbose_flag:
                print(f'{_PREFIX} ⚠ Could not access diffusion model for reference conditioning preprocessing: {exc}')

        cfg: Dict[str, Any] = {
            'rf_mode': str(rf_mode),
            'gamma': float(gamma),
            'gamma_curve': float(gamma_curve),
            'norm_strength': float(norm_strength),
            'pmi_alpha': float(pmi_alpha),
            'seed': 42,
            'verbose': verbose_flag,
            'apply_model_output': 'comfy_denoised_x0',
            'model_info': model_info,
        }
        state: Dict[str, Any] = {
            'cache': {0.0: ref_clean.detach().to(device='cpu').clone()},
            'eps': None,
            'prev_z': None,
            'prev_sigma': None,
            'run_count': 0,
            'sampler_sigmas': None,
            'schedule_built': False,
            'schedule_sorted': None,
            'persistent_cache_key': None,
            'persistent_cache_hit': False,
            'preview_callback': None,
            'wrapper_calls': 0,
            'model_info': model_info,
            'last_sigma': None,
            'last_cond_mode': None,
            'last_cache_lookup': None,
            'last_error': None,
        }
        debug_store = _rf_new_debug_store()
        debug_store['cache'] = state['cache']
        debug_store['parameterization'] = detected_param
        debug_store['apply_model_output'] = cfg['apply_model_output']
        debug_store['model_info'] = model_info

        # Normal LATENT output: samples stay a latent tensor; extra keys carry RF metadata.
        rf_latent: Dict[str, Any] = dict(reference_latent)
        rf_latent['samples'] = reference_latent['samples']
        rf_latent['untwist_rf_config'] = cfg
        rf_latent['untwist_rf_state'] = state
        rf_latent['untwist_rf_cache'] = state['cache']
        rf_latent['untwist_rf_sigmas'] = None
        rf_latent['untwist_rf_mode'] = str(rf_mode)
        rf_latent['untwist_rf_seed'] = 42
        rf_latent['untwist_rf_parameterization'] = detected_param
        rf_latent['untwist_rf_apply_model_output'] = cfg['apply_model_output']
        rf_latent['untwist_rf_model_info'] = model_info
        rf_latent['untwist_ref_clean'] = ref_clean.detach().to(device='cpu').clone()
        rf_latent['untwist_ref_conditioning'] = ref_conditioning

        model_clone = model.clone()
        setattr(model_clone, '_untwisting_rope_rf_debug', debug_store)
        setattr(model_clone, '_untwisting_rope_rf_state', state)
        setattr(model_clone, '_untwisting_rope_rf_config', cfg)

        def sampler_sample_wrapper(executor, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            found = _coerce_sigma_sequence(sigmas)
            if found is not None:
                state['sampler_sigmas'] = found
                state['schedule_built'] = False
                state['schedule_sorted'] = None
                state['persistent_cache_key'] = None
                state['persistent_cache_hit'] = False
                state['cache'] = {0.0: ref_clean.detach().to(device='cpu').clone()}
                state['eps'] = None
                state['run_count'] = int(state.get('run_count', 0)) + 1
                try:
                    state['preview_callback'] = _rf_make_preview_callback(model_clone, max(1, len(found) - 1))
                except Exception:
                    state['preview_callback'] = None

                rf_latent['untwist_rf_cache'] = state['cache']
                rf_latent['untwist_rf_sigmas'] = list(found)
                rf_latent['untwist_rf_state'] = state

                debug_store['cache'] = state['cache']
                debug_store['sampler_sigmas'] = list(found)
                debug_store['built_sigmas'] = None
                debug_store['run_count'] = int(state['run_count'])
                debug_store['persistent_cache_key'] = None
                debug_store['persistent_cache_hit'] = False
                debug_store['parameterization'] = rf_latent.get('untwist_rf_parameterization', 'unknown')
                if verbose_flag:
                    print(
                        f'{_PREFIX} RFInversion sampler_sample: captured {len(found)} sigmas  '
                        f'run={state["run_count"]}  seed=42'
                    )
            return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)

        model_clone.model_options = _clone_model_options(model_clone.model_options)
        comfy.patcher_extension.add_wrapper(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            sampler_sample_wrapper,
            model_clone.model_options,
            is_model_options=True,
        )

        # RFInversion must be able to run by itself. The original code only
        # captured sampler sigmas here; the trajectory was built later inside
        # UntwistingRoPE.patch, which is architecture-specific. This wrapper
        # builds the RF cache during the normal sampler model calls and then
        # returns the original model prediction unchanged.
        old_model_function_wrapper = model_clone.model_options.get('model_function_wrapper', None)
        rf_runtime_stats = _RuntimeStats(verbose=False, rf_verbose=verbose_flag)
        rf_runtime_stats.parameterization = detected_param

        def rf_model_function_wrapper(apply_model: Callable, args: Dict[str, Any]) -> torch.Tensor:
            state['wrapper_calls'] = int(state.get('wrapper_calls', 0)) + 1
            call_n = int(state['wrapper_calls'])
            debug_store['wrapper_calls'] = call_n

            input_x = args.get('input', None)
            timestep = args.get('timestep', None)
            c_in = args.get('c', {})
            c = c_in.copy() if isinstance(c_in, dict) else {}
            sigma = _sigma_from_timestep(timestep) if torch.is_tensor(timestep) else 1.0
            sigma_key = round(float(sigma), 6)
            state['last_sigma'] = sigma_key
            debug_store['last_sigma'] = sigma_key

            try:
                if not torch.is_tensor(input_x):
                    raise RuntimeError('RFInversion wrapper received a non-tensor input.')

                target_b = int(input_x.shape[0])
                rf_ref_clean = _repeat_to_batch(ref_clean.to(device=input_x.device, dtype=input_x.dtype), target_b)
                sampler_sigmas = state.get('sampler_sigmas', None)

                # Build the full sampler-grid RF trajectory once per sampler run.
                if not state.get('schedule_built', False) and sampler_sigmas is not None:
                    effective_ref_conditioning, adapter_ref_status = _prepare_reference_conditioning_for_adapter(
                        adapter, ref_conditioning, dm_for_ref, input_x.device,
                        c.get('c_crossattn').dtype if torch.is_tensor(c.get('c_crossattn', None)) else input_x.dtype,
                        rf_runtime_stats, label='RFInversion',
                    )
                    rf_kwargs, rf_cond_mode = _build_rf_conditioning_kwargs(c, effective_ref_conditioning, target_b)
                    rf_cond_mode = _append_conditioning_status(rf_cond_mode, adapter_ref_status)
                    state['last_cond_mode'] = rf_cond_mode
                    debug_store['last_cond_mode'] = rf_cond_mode

                    cache_key = _make_rf_persistent_key(
                        ref_clean=ref_clean.detach().to(device='cpu'),
                        ref_conditioning=ref_conditioning,
                        sampler_sigmas=list(sampler_sigmas),
                        target_b=target_b,
                        rf_mode=rf_mode,
                        gamma=gamma,
                        gamma_curve=gamma_curve,
                        norm_strength=norm_strength,
                        cond_mode=rf_cond_mode,
                        pmi_alpha=pmi_alpha,
                    )

                    if verbose_flag:
                        print(f'{_PREFIX}   RF build requested from RFInversion wrapper')
                        print(f'{_PREFIX}   {_rf_sequence_summary("sampler_sigmas", sampler_sigmas)}')
                        print(f'{_PREFIX}   target_b={target_b} cond_mode={rf_cond_mode} cache_key={cache_key[:12]}')
                        print(f'{_PREFIX}   rf_ref_clean: {_rf_tensor_summary("rf_ref_clean", rf_ref_clean)}')
                        print(f'{_PREFIX}   rf_kwargs summary: {_rf_brief_obj(rf_kwargs)}')

                    cached_entry = _RF_PERSISTENT_TRAJECTORY_CACHE.get(cache_key)
                    if cached_entry is not None:
                        built_cache = _cache_to_device(cached_entry['cache'], input_x.device, input_x.dtype)
                        eps = cached_entry['eps'].to(device=input_x.device, dtype=input_x.dtype)
                        sorted_sigmas = list(cached_entry['built_sigmas'])
                        state['persistent_cache_hit'] = True
                        if verbose_flag:
                            print(f'{_PREFIX}   RF persistent cache HIT key={cache_key[:12]} cache_items={len(built_cache)}')
                    else:
                        state['persistent_cache_hit'] = False
                        preview_callback = state.get('preview_callback', None)
                        if preview_callback is None:
                            preview_callback = _rf_make_preview_callback(model_clone, max(1, len(list(sampler_sigmas)) - 1))
                            state['preview_callback'] = preview_callback
                        if verbose_flag:
                            print(f'{_PREFIX}   RF persistent cache MISS key={cache_key[:12]} → building now')
                        built_cache, eps, sorted_sigmas = _rf_build_cache_from_sampler_sigmas(
                            ref_clean=rf_ref_clean,
                            sampler_sigmas=list(sampler_sigmas),
                            apply_model_fn=apply_model,
                            base_model_kwargs=rf_kwargs,
                            gamma=gamma,
                            seed=42,
                            stats=rf_runtime_stats,
                            eps=state['eps'].to(device=input_x.device, dtype=input_x.dtype)
                                if torch.is_tensor(state.get('eps', None)) else None,
                            rf_mode=rf_mode,
                            gamma_curve=gamma_curve,
                            norm_strength=norm_strength,
                            pmi_alpha=pmi_alpha,
                            preview_callback=preview_callback,
                        )
                        _put_persistent_rf_cache(cache_key, {
                            'cache': _cache_to_cpu(built_cache),
                            'eps': eps.detach().to(device='cpu').clone(),
                            'built_sigmas': list(sorted_sigmas),
                        })

                    state['cache'] = built_cache
                    state['eps'] = eps.detach().clone()
                    state['schedule_sorted'] = list(sorted_sigmas)
                    state['schedule_built'] = True
                    state['persistent_cache_key'] = cache_key
                    rf_latent['untwist_rf_cache'] = _cache_to_cpu(built_cache)
                    rf_latent['untwist_rf_eps'] = eps.detach().to(device='cpu').clone()
                    rf_latent['untwist_rf_sigmas'] = list(sorted_sigmas)
                    rf_latent['untwist_rf_state'] = state

                    debug_store['cache'] = state['cache']
                    debug_store['sampler_sigmas'] = list(sampler_sigmas)
                    debug_store['built_sigmas'] = list(sorted_sigmas)
                    debug_store['persistent_cache_key'] = cache_key
                    debug_store['persistent_cache_hit'] = bool(state.get('persistent_cache_hit', False))
                    debug_store['parameterization'] = detected_param
                    debug_store['apply_model_output'] = cfg['apply_model_output']
                    debug_store['model_info'] = model_info

                    rf_sanity = _rf_stability_summary(rf_ref_clean, eps, built_cache, list(sorted_sigmas))
                    state['last_stability_summary'] = rf_sanity
                    rf_latent['untwist_rf_stability_summary'] = rf_sanity
                    debug_store['stability_summary'] = rf_sanity

                    if verbose_flag:
                        print(
                            f'{_PREFIX}   RF build complete: cache_items={len(built_cache)} '
                            f'built_sigmas={len(sorted_sigmas)} eps={_rf_tensor_summary("eps", eps)}'
                        )
                        _rf_print_stability_summary(rf_sanity)

                elif not state.get('schedule_built', False) and sampler_sigmas is None:
                    # This should be rare because SAMPLER_SAMPLE normally runs before
                    # model calls. Keep it as an explicit diagnostic fallback.
                    effective_ref_conditioning, adapter_ref_status = _prepare_reference_conditioning_for_adapter(
                        adapter, ref_conditioning, dm_for_ref, input_x.device,
                        c.get('c_crossattn').dtype if torch.is_tensor(c.get('c_crossattn', None)) else input_x.dtype,
                        rf_runtime_stats, label='RFInversionFallback',
                    )
                    rf_kwargs, rf_cond_mode = _build_rf_conditioning_kwargs(c, effective_ref_conditioning, target_b)
                    rf_cond_mode = _append_conditioning_status(rf_cond_mode, adapter_ref_status)
                    state['last_cond_mode'] = rf_cond_mode
                    debug_store['last_cond_mode'] = rf_cond_mode
                    if verbose_flag:
                        print(f'{_PREFIX}   ⚠ No sampler sigmas captured yet; direct one-step RF fallback for σ={sigma:.6f}')
                    z_sigma, eps = _rf_increment_reference_one_step(
                        z_prev=rf_ref_clean,
                        sigma_prev=0.0,
                        sigma_cur=sigma,
                        apply_model_fn=apply_model,
                        base_model_kwargs=rf_kwargs,
                        gamma=gamma,
                        seed=42,
                        stats=rf_runtime_stats,
                        eps=state['eps'].to(device=input_x.device, dtype=input_x.dtype)
                            if torch.is_tensor(state.get('eps', None)) else None,
                        rf_mode=rf_mode,
                        gamma_curve=gamma_curve,
                        norm_strength=norm_strength,
                        preview_callback=state.get('preview_callback', None),
                    )
                    cache = state.get('cache') if isinstance(state.get('cache'), dict) else {}
                    cache[sigma_key] = z_sigma.detach().clone()
                    state['cache'] = cache
                    state['eps'] = eps.detach().clone()
                    rf_latent['untwist_rf_cache'] = _cache_to_cpu(cache)
                    rf_latent['untwist_rf_eps'] = eps.detach().to(device='cpu').clone()
                    debug_store['cache'] = state['cache']
                    if torch.is_tensor(state.get('eps', None)):
                        rf_sanity = _rf_stability_summary(rf_ref_clean, state['eps'], cache, sorted(cache.keys()))
                        state['last_stability_summary'] = rf_sanity
                        rf_latent['untwist_rf_stability_summary'] = rf_sanity
                        debug_store['stability_summary'] = rf_sanity
                        if verbose_flag:
                            _rf_print_stability_summary(rf_sanity)

                cache = state.get('cache') if isinstance(state.get('cache'), dict) else {}
                cached = cache.get(sigma_key, None)
                cache_lookup = 'exact' if cached is not None else 'missing'
                if cached is None:
                    keys = [k for k in cache.keys() if isinstance(k, float)]
                    if keys:
                        nearest = min(keys, key=lambda k: abs(k - sigma_key))
                        cached = cache.get(nearest)
                        cache_lookup = f'nearest:{nearest:.6f}:absdiff={abs(nearest - sigma_key):.6f}'
                state['last_cache_lookup'] = cache_lookup
                debug_store['last_cache_lookup'] = cache_lookup

                if verbose_flag:
                    pass

            except Exception as exc:
                state['last_error'] = repr(exc)
                debug_store['last_error'] = repr(exc)
                print(f'{_PREFIX} ⚠ RFInversion standalone wrapper failed; sampling will continue unchanged: {exc}')
                if verbose_flag:
                    print(traceback.format_exc())

            if old_model_function_wrapper is not None:
                return old_model_function_wrapper(apply_model, args)
            return apply_model(args['input'], args['timestep'], **args['c'])

        model_clone.set_model_unet_function_wrapper(rf_model_function_wrapper)

        if verbose_flag:
            print(f'\n{_PREFIX} ═══════════════════════════════════════')
            print(f'{_PREFIX} RF INVERSION PREPARED')
            print(f'{_PREFIX} ═══════════════════════════════════════')
            print(f'{_PREFIX}   mode          : {rf_mode}')
            print(f'{_PREFIX}   gamma         : {float(gamma):.4f}')
            print(f'{_PREFIX}   gamma_curve   : {float(gamma_curve):.3f}')
            print(f'{_PREFIX}   norm_strength : {float(norm_strength):.3f}')
            print(f'{_PREFIX}   pmi_alpha     : {float(pmi_alpha):.3f}')
            print(f'{_PREFIX}   seed          : 42 (internal fixed noise seed)')
            print(f'{_PREFIX}   schedule      : captured from sampler at runtime; no SIGMAS input')
            print(f'{_PREFIX}   output        : normal LATENT with RF metadata')
            print(f'{_PREFIX}   wrapper       : standalone RF cache builder installed on MODEL')
            print(f'{_PREFIX}   diagnostics   : verbose=True prints per-call/cache/conditioning details')
            _rf_print_model_identity(f'{_PREFIX}   RFInversion', model_info)
            print(f'{_PREFIX} ═══════════════════════════════════════\n')

        return (model_clone, rf_latent)

class UntwistingRoPE:
    CATEGORY = 'model_patches/Untwisting RoPE'
    RETURN_TYPES = ('MODEL',)
    RETURN_NAMES = ('model',)
    FUNCTION = 'patch'
    DESCRIPTION = (
        'Patches supported attention/RoPE modules and uses the RFInversion LATENT trajectory. '
        'RF inversion settings live on the LATENT; the sampler sigma schedule is captured internally.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'beta': ('FLOAT', {'default': 50.0, 'min': 0.01, 'max': 100.0, 'step': 0.01}),
                'high_scale_start': ('FLOAT', {'default': 1.0, 'min': -4.0, 'max': 8.0, 'step': 0.01}),
                'high_scale_end': ('FLOAT', {'default': 0.00, 'min': -4.0, 'max': 8.0, 'step': 0.01}),
                'low_scale_start': ('FLOAT', {'default': 1.0, 'min': -4.0, 'max': 8.0, 'step': 0.01}),
                'low_scale_end': ('FLOAT', {'default': 3.0, 'min': -4.0, 'max': 8.0, 'step': 0.01}),
                'start_block': ('INT', {'default': 0, 'min': 0, 'max': 999, 'step': 1}),
                'end_block': ('INT', {'default': 999, 'min': 0, 'max': 999, 'step': 1}),
                'adain_strength': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'verbose': ('BOOLEAN', {'default': False}),
            },
            'optional': {
                'rf_inversion': ('LATENT',),
            },
        }

    def patch(
        self,
        model,
        beta: float,
        high_scale_start: float,
        high_scale_end: float,
        low_scale_start: float,
        low_scale_end: float,
        start_block: int,
        end_block: int,
        adain_strength: float,
        verbose: bool = False,
        rf_inversion: Optional[Dict[str, Any]] = None,
    ):
        rf_active, rf_cfg, rf_state, ref_clean_cpu, ref_conditioning, rf_source = _rf_latent_get_config(rf_inversion)
        node_verbose = _coerce_bool(verbose)
        rf_verbose = _coerce_bool(rf_cfg.get('verbose', False))
        stats = _RuntimeStats(verbose=node_verbose, rf_verbose=rf_verbose)
        debug_store = _rf_new_debug_store()

        rf_mode = str(rf_cfg.get('rf_mode', 'fireflow'))
        gamma = float(rf_cfg.get('gamma', 0.3))
        gamma_curve = float(rf_cfg.get('gamma_curve', 0.0))
        norm_strength = float(rf_cfg.get('norm_strength', 0.0))
        pmi_alpha = float(rf_cfg.get('pmi_alpha', 0.4))
        seed = int(rf_cfg.get('seed', 42))

        if rf_active:
            stats.rf_sigma_cache = rf_state.get('cache', {}) if isinstance(rf_state.get('cache', {}), dict) else {}
            stats.rf_schedule_built = bool(rf_state.get('schedule_built', False))
            stats.parameterization = str(rf_inversion.get('untwist_rf_parameterization', 'unknown')) if isinstance(rf_inversion, dict) else 'unknown'
            debug_store['cache'] = stats.rf_sigma_cache
            debug_store['sampler_sigmas'] = list(rf_state.get('sampler_sigmas') or []) if isinstance(rf_state, dict) else []
            debug_store['built_sigmas'] = list(rf_state.get('schedule_sorted') or []) if isinstance(rf_state, dict) else []
            debug_store['parameterization'] = stats.parameterization

        _vprint(stats, f'\n{_PREFIX} ═══════════════════════════════════════')
        _vprint(stats, f'{_PREFIX} PATCH START  (split nodes: RFInversion + UntwistingRoPE)')
        _vprint(stats, f'{_PREFIX} ═══════════════════════════════════════')
        _vprint(stats, f'{_PREFIX} beta={beta}')
        _vprint(stats, f'{_PREFIX} high_scale: {high_scale_start:.3f} → {high_scale_end:.3f}')
        _vprint(stats, f'{_PREFIX} low_scale:  {low_scale_start:.3f} → {low_scale_end:.3f}')
        _vprint(stats,
            f'{_PREFIX} blocks [{start_block}, {end_block}]  '
            f'adain={adain_strength:.2f}'
        )
        _vprint(stats, f'{_PREFIX} RF latent connected: {rf_active}  source={rf_source}')
        if rf_active:
            _vprint(stats,
                f'{_PREFIX} RF trajectory: mode={rf_mode}  gamma={gamma}  '
                f'gamma_curve={gamma_curve:.3f}  '
                f'norm_strength={norm_strength}  pmi_alpha={pmi_alpha}  seed={seed}'
            )
            _vprint(stats, f'{_PREFIX} RF schedule: captured from sampler at runtime; no SIGMAS input')

        model_clone = model.clone()
        setattr(model_clone, '_untwisting_rope_rf_debug', debug_store)
        if rf_active:
            setattr(model_clone, '_untwisting_rope_rf_state', rf_state)
            setattr(model_clone, '_untwisting_rope_rf_config', rf_cfg)

        model_info = _rf_model_identity(model_clone)
        adapter = _select_model_adapter(model_clone, model_info)
        dm = _safe_get_diffusion_model(model_clone, adapter)
        _vprint(stats, f'{_PREFIX} Diffusion model type: {type(dm).__name__}')

        global _ACTIVE_MODEL_ADAPTER
        previous_adapter = _ACTIVE_MODEL_ADAPTER
        _ACTIVE_MODEL_ADAPTER = adapter
        try:
            patch_fn = getattr(adapter, 'patch_attention_modules', None)
            if callable(patch_fn):
                patch_fn(dm, stats, _adapter_helpers())
            else:
                _patch_context_refiner_mask_modules(dm, stats)
                _patch_patchify_and_embed(dm, stats)
                _patch_joint_attention_modules(dm, stats)
        finally:
            _ACTIVE_MODEL_ADAPTER = previous_adapter

        old_wrapper = model_clone.model_options.get('model_function_wrapper', None)

        def model_function_wrapper(apply_model: Callable, args: Dict[str, Any]) -> torch.Tensor:
            stats.wrapper_calls += 1
            call_n = stats.wrapper_calls

            input_x = args['input']
            timestep = args['timestep']
            c = args['c'].copy()
            cond_or_uncond = args.get('cond_or_uncond', None)
            to = c.get('transformer_options', {}).copy()

            sigma = _sigma_from_timestep(timestep)
            progress = _sigma_to_progress(timestep)
            target_b = int(input_x.shape[0])

            cfg: Dict[str, Any] = {
                'enabled': True,
                'beta': float(beta),
                'high_scale_start': float(high_scale_start),
                'high_scale_end': float(high_scale_end),
                'low_scale_start': float(low_scale_start),
                'low_scale_end': float(low_scale_end),
                'start_block': int(start_block),
                'end_block': int(end_block),
                'apply_adain': True,
                'adain_strength': float(adain_strength),
                'cross_batch_target_batch': target_b if rf_active else 0,
                'progress': progress,
            }
            default_cfg = getattr(adapter, 'default_runtime_cfg', lambda _dm=None: {})
            cfg.update(default_cfg(dm))
            to[_TRANSFORMER_CONFIG_KEY] = cfg

            input_for_model = input_x
            timestep_for_model = timestep
            ref_noisy = None
            sigma_key = round(float(sigma), 6)
            rf_cache_hit = False
            rf_cond_mode = 'not-connected'
            ref_mode = 'target-only'
            should_print = _coerce_bool(getattr(stats, 'rf_verbose', False))

            if rf_active and torch.is_tensor(ref_clean_cpu):
                try:
                    ref_clean = ref_clean_cpu.to(device=input_x.device, dtype=input_x.dtype)
                    ref = _repeat_to_batch(ref_clean, target_b)

                    if not rf_state.get('schedule_built', False) and rf_state.get('sampler_sigmas', None) is not None:
                        effective_ref_conditioning, adapter_ref_status = _prepare_reference_conditioning_for_adapter(
                            adapter, ref_conditioning, dm, input_x.device,
                            c.get('c_crossattn').dtype if torch.is_tensor(c.get('c_crossattn', None)) else input_x.dtype,
                            stats, label='UntwistingRoPE',
                        )
                        rf_kwargs, rf_cond_mode = _build_rf_conditioning_kwargs(c, effective_ref_conditioning, target_b)
                        rf_cond_mode = _append_conditioning_status(rf_cond_mode, adapter_ref_status)
                        rf_ref_clean = _repeat_to_batch(ref_clean, target_b)
                        sampler_sigmas = list(rf_state['sampler_sigmas'])
                        cache_key = _make_rf_persistent_key(
                            ref_clean=ref_clean_cpu.detach().to(device='cpu'),
                            ref_conditioning=ref_conditioning,
                            sampler_sigmas=sampler_sigmas,
                            target_b=target_b,
                            rf_mode=rf_mode,
                            gamma=gamma,
                            gamma_curve=gamma_curve,
                            norm_strength=norm_strength,
                            cond_mode=rf_cond_mode,
                            pmi_alpha=pmi_alpha,
                        )
                        cached_entry = _RF_PERSISTENT_TRAJECTORY_CACHE.get(cache_key)
                        if cached_entry is not None:
                            built_cache = _cache_to_device(cached_entry['cache'], input_x.device, input_x.dtype)
                            eps = cached_entry['eps'].to(device=input_x.device, dtype=input_x.dtype)
                            sorted_sigmas = list(cached_entry['built_sigmas'])
                            _rf_vprint(stats, f'{_PREFIX} RFInversion persistent cache HIT: key={cache_key[:12]}  cache={len(built_cache)}')
                            ref_mode = 'RF sampler-sigma trajectory (persistent-cache hit)'
                            rf_state['persistent_cache_hit'] = True
                        else:
                            _rf_vprint(stats, f'{_PREFIX} RFInversion persistent cache MISS: key={cache_key[:12]}  building trajectory')
                            preview_callback = rf_state.get('preview_callback', None)
                            if preview_callback is None:
                                preview_callback = _rf_make_preview_callback(model_clone, max(1, len(sampler_sigmas) - 1))
                                rf_state['preview_callback'] = preview_callback
                            built_cache, eps, sorted_sigmas = _rf_build_cache_from_sampler_sigmas(
                                ref_clean=rf_ref_clean,
                                sampler_sigmas=sampler_sigmas,
                                apply_model_fn=apply_model,
                                base_model_kwargs=rf_kwargs,
                                gamma=gamma,
                                seed=seed,
                                stats=stats,
                                eps=rf_state['eps'].to(device=input_x.device, dtype=input_x.dtype)
                                    if torch.is_tensor(rf_state.get('eps', None)) else None,
                                rf_mode=rf_mode,
                                gamma_curve=gamma_curve,
                                    norm_strength=norm_strength,
                                pmi_alpha=pmi_alpha,
                                preview_callback=preview_callback,
                            )
                            _put_persistent_rf_cache(cache_key, {
                                'cache': _cache_to_cpu(built_cache),
                                'eps': eps.detach().to(device='cpu').clone(),
                                'built_sigmas': list(sorted_sigmas),
                            })
                            ref_mode = 'RF sampler-sigma trajectory (built)'
                            rf_state['persistent_cache_hit'] = False

                        rf_state['cache'] = built_cache
                        rf_state['eps'] = eps.detach().clone()
                        rf_state['schedule_sorted'] = sorted_sigmas
                        rf_state['schedule_built'] = True
                        rf_state['persistent_cache_key'] = cache_key
                        stats.rf_sigma_cache = rf_state['cache']
                        stats.rf_eps = rf_state['eps']
                        stats.rf_schedule_built = True
                        stats.rf_step_count = max(0, len(sorted_sigmas) - 1)

                        if isinstance(rf_inversion, dict):
                            rf_inversion['untwist_rf_cache'] = _cache_to_cpu(built_cache)
                            rf_inversion['untwist_rf_eps'] = eps.detach().to(device='cpu').clone()
                            rf_inversion['untwist_rf_sigmas'] = list(sorted_sigmas)
                            rf_inversion['untwist_rf_state'] = rf_state

                        debug_store['cache'] = rf_state['cache']
                        debug_store['sampler_sigmas'] = list(rf_state.get('sampler_sigmas') or [])
                        debug_store['built_sigmas'] = list(sorted_sigmas)
                        debug_store['run_count'] = int(rf_state.get('run_count', 0))
                        debug_store['persistent_cache_key'] = cache_key
                        debug_store['persistent_cache_hit'] = bool(rf_state.get('persistent_cache_hit', False))
                        debug_store['parameterization'] = stats.parameterization
                    elif rf_state.get('schedule_built', False):
                        ref_mode = 'RF sampler-sigma trajectory (cached)'
                    else:
                        # Fallback: no sampler wrapper triggered. This preserves original behavior.
                        effective_ref_conditioning, adapter_ref_status = _prepare_reference_conditioning_for_adapter(
                            adapter, ref_conditioning, dm, input_x.device,
                            c.get('c_crossattn').dtype if torch.is_tensor(c.get('c_crossattn', None)) else input_x.dtype,
                            stats, label='UntwistingRoPEFallback',
                        )
                        rf_kwargs, rf_cond_mode = _build_rf_conditioning_kwargs(c, effective_ref_conditioning, target_b)
                        rf_cond_mode = _append_conditioning_status(rf_cond_mode, adapter_ref_status)
                        rf_ref_clean = _repeat_to_batch(ref_clean, target_b)
                        preview_callback = rf_state.get('preview_callback', None)
                        if preview_callback is None:
                            preview_callback = _rf_make_preview_callback(model_clone, 1)
                            rf_state['preview_callback'] = preview_callback
                        z_sigma, eps = _rf_increment_reference_one_step(
                            z_prev=rf_ref_clean,
                            sigma_prev=0.0,
                            sigma_cur=sigma,
                            apply_model_fn=apply_model,
                            base_model_kwargs=rf_kwargs,
                            gamma=gamma,
                            seed=seed,
                            stats=stats,
                            eps=rf_state['eps'].to(device=input_x.device, dtype=input_x.dtype)
                                if torch.is_tensor(rf_state.get('eps', None)) else None,
                            rf_mode=rf_mode,
                            gamma_curve=gamma_curve,
                            norm_strength=norm_strength,
                            preview_callback=preview_callback,
                        )
                        rf_state['eps'] = eps.detach().clone()
                        cache = rf_state.get('cache') if isinstance(rf_state.get('cache'), dict) else {}
                        cache[sigma_key] = z_sigma.detach().clone()
                        rf_state['cache'] = cache
                        stats.rf_eps = rf_state['eps']
                        stats.rf_sigma_cache = rf_state['cache']
                        ref_mode = 'RF direct fallback (no sampler wrapper)'

                    cache = rf_state.get('cache') if isinstance(rf_state.get('cache'), dict) else {}
                    cached = cache.get(sigma_key, None)
                    if cached is None:
                        keys = [k for k in cache.keys() if isinstance(k, float)]
                        if keys:
                            nearest = min(keys, key=lambda k: abs(k - sigma_key))
                            cached = cache[nearest]
                            ref_mode += f' nearest({nearest:.6f})'
                        else:
                            cached = ref
                    else:
                        rf_cache_hit = True

                    ref_noisy = _repeat_to_batch(cached.to(device=input_x.device, dtype=input_x.dtype), target_b)

                    if should_print:
                        print(
                            f'{_PREFIX}   input_x   mean={input_x.mean().item():.4f}  std={input_x.std().item():.4f}\n'
                            f'{_PREFIX}   ref_noisy mean={ref_noisy.mean().item():.4f}  std={ref_noisy.std().item():.4f}  [{ref_mode}]'
                        )

                    if ref_noisy.shape[-2:] == input_x.shape[-2:]:
                        input_for_model = torch.cat([input_x, ref_noisy], dim=0)
                        try:
                            if (torch.is_tensor(timestep)
                                    and timestep.ndim > 0
                                    and int(timestep.shape[0]) == target_b):
                                timestep_for_model = torch.cat([timestep, timestep], dim=0)
                            else:
                                timestep_for_model = _repeat_to_batch(timestep, target_b * 2)
                        except Exception:
                            timestep_for_model = timestep

                        effective_ref_conditioning, adapter_ref_status = _prepare_reference_conditioning_for_adapter(
                            adapter, ref_conditioning, dm, input_x.device,
                            c.get('c_crossattn').dtype if torch.is_tensor(c.get('c_crossattn', None)) else input_x.dtype,
                            stats, label='UntwistingRoPEMerge',
                        )
                        c, forced_cap_mask = _merge_reference_conditioning_into_c(c, effective_ref_conditioning, target_b)
                        cfg['adapter_ref_conditioning_status'] = adapter_ref_status
                        cfg['forced_cap_mask'] = forced_cap_mask.to(device=input_x.device)
                        cfg['cross_batch_target_batch'] = target_b

                        try:
                            if isinstance(cond_or_uncond, list):
                                cond_or_uncond = cond_or_uncond + cond_or_uncond
                        except Exception:
                            pass
                    else:
                        print(
                            f'{_PREFIX} ⚠ [call={call_n}] Spatial mismatch '
                            f'input_x={tuple(input_x.shape[-2:])} '
                            f'ref_noisy={tuple(ref_noisy.shape[-2:])} '
                            f'→ target-only fallback'
                        )
                        cfg['enabled'] = False
                        cfg['cross_batch_target_batch'] = 0
                        ref_noisy = None
                except Exception as exc:
                    print(f'{_PREFIX} ⚠ UntwistingRoPE RF latent fallback to target-only: {exc}')
                    cfg['enabled'] = False
                    cfg['cross_batch_target_batch'] = 0
                    ref_noisy = None

            c['transformer_options'] = to

            if old_wrapper is not None:
                raw_result = old_wrapper(apply_model, {
                    'input': input_for_model,
                    'timestep': timestep_for_model,
                    'c': c,
                    'cond_or_uncond': cond_or_uncond,
                })
            else:
                raw_result = apply_model(input_for_model, timestep_for_model, **c)

            if (rf_active
                    and ref_noisy is not None
                    and torch.is_tensor(raw_result)
                    and raw_result.shape[0] >= target_b * 2):
                target_pred = raw_result[:target_b]
                ref_pred = raw_result[target_b:target_b * 2]

                try:
                    ref_xsigma = ref_noisy[:target_b]
                    debug_store['pred_cache'][sigma_key] = ref_pred[:1].detach().clone()
                    debug_store['xhat_cache'][sigma_key] = (ref_xsigma - float(sigma) * ref_pred)[:1].detach().clone()
                    debug_store['xhat_plus_cache'][sigma_key] = (ref_xsigma + float(sigma) * ref_pred)[:1].detach().clone()
                except Exception as exc:
                    print(f'{_PREFIX} ⚠ Failed to cache RF debug latents at σ={float(sigma):.6f}: {exc}')

                if should_print:
                    pass
                return target_pred

            return raw_result

        model_clone.model_options = _clone_model_options(model_clone.model_options)
        model_clone.set_model_unet_function_wrapper(model_function_wrapper)

        _vprint(stats, f'\n{_PREFIX} ═══════════════════════════════════════')
        _vprint(stats, f'{_PREFIX} PATCH COMPLETE')
        if rf_active:
            _vprint(stats, f'{_PREFIX}   RF input      : LATENT from RFInversion')
            _vprint(stats, f'{_PREFIX}   RF schedule   : captured internally by RFInversion model wrapper')
            _vprint(stats, f'{_PREFIX}   RF preview    : emitted while building inversion trajectory')
            uses_kv = getattr(adapter, 'uses_reference_branch_kv', lambda: False)
            if bool(uses_kv()):
                _vprint(stats, f'{_PREFIX}   K/V           : reference branch contributes K and V; only K is untwisted')
        else:
            _vprint(stats, f'{_PREFIX}   RF input      : not connected')
            _vprint(stats, f'{_PREFIX}   Mode          : target-only attention patch')
        _vprint(stats, f'{_PREFIX}   Output: target prediction returned unchanged')
        _vprint(stats, f'{_PREFIX} ═══════════════════════════════════════\n')

        return (model_clone,)

NODE_CLASS_MAPPINGS = {
    'RFInversion': RFInversion,
    'UntwistingRoPE': UntwistingRoPE,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'RFInversion': 'RF Inversion',
    'UntwistingRoPE': 'Untwisting RoPE',
}
