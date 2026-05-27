from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import torch

_PREFIX = '[UntwistingRoPE]'
_RF_PREFIX = '[RFInversion]'

def _rf_prefix(stats: Optional[Any] = None) -> str:
    if stats is None:
        return _RF_PREFIX
    prefix = getattr(stats, 'rf_prefix', None)
    if prefix is None:
        return _RF_PREFIX
    if not isinstance(prefix, str) or not prefix:
        raise RuntimeError(f'Invalid RF prefix on stats: {prefix!r}')
    return prefix

def _coerce_bool(value: Any) -> bool:
    """Robust boolean parsing for ComfyUI values that may arrive as bools or strings."""
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on', 'y', 't')
    return bool(value)

class _RuntimeStats:
    def __init__(self, verbose: bool = False, rf_verbose: bool = False) -> None:
        # verbose controls UntwistingRoPE patch/attention logs.
        # rf_verbose controls RFInversion trajectory/wrapper logs.
        self.verbose: bool = _coerce_bool(verbose)
        self.rf_verbose: bool = _coerce_bool(rf_verbose)
        self.rf_prefix: str = _RF_PREFIX
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
        raise RuntimeError(f'{_RF_PREFIX} tensor summary failed for {name}; strict mode refuses to hide diagnostic failure: {exc}') from exc

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
        raise RuntimeError(f'{_RF_PREFIX} tensor stats failed; strict mode refuses to hide diagnostic failure: {exc}') from exc
    return out

def _rf_scalar_fmt(value: Any, digits: int = 6) -> str:
    try:
        if value is None:
            return 'n/a'
        value = float(value)
        if not math.isfinite(value):
            return str(value)
        return f'{value:.{digits}f}'
    except Exception as exc:
        raise RuntimeError(f'{_RF_PREFIX} scalar formatting failed for value={value!r}: {exc}') from exc

def _rf_tensor_mae(a: Any, b: Any) -> Optional[float]:
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return None
    try:
        aa = a.detach().float().to(device='cpu')
        bb = b.detach().float().to(device='cpu')
        if aa.shape != bb.shape or aa.numel() == 0:
            return None
        return float((aa - bb).abs().mean().item())
    except Exception as exc:
        raise RuntimeError(f'{_RF_PREFIX} tensor MAE failed; strict mode refuses to hide diagnostic failure: {exc}') from exc

def _rf_tensor_rmse(a: Any, b: Any) -> Optional[float]:
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return None
    try:
        aa = a.detach().float().to(device='cpu')
        bb = b.detach().float().to(device='cpu')
        if aa.shape != bb.shape or aa.numel() == 0:
            return None
        return float((aa - bb).pow(2).mean().sqrt().item())
    except Exception as exc:
        raise RuntimeError(f'{_RF_PREFIX} tensor RMSE failed; strict mode refuses to hide diagnostic failure: {exc}') from exc

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
    except Exception as exc:
        raise RuntimeError(f'{_RF_PREFIX} tensor cosine failed; strict mode refuses to hide diagnostic failure: {exc}') from exc

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
    except Exception as exc:
        raise RuntimeError(f'{_RF_PREFIX} diagnostic-level calculation failed; strict mode refuses to downgrade to WARN: {exc}') from exc
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
            except Exception as exc:
                raise RuntimeError(f'{_RF_PREFIX} Δz diagnostic failed; strict mode refuses to skip it: {exc}') from exc
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
    except Exception as exc:
        raise RuntimeError(f'{_RF_PREFIX} final/eps std-ratio diagnostic failed; strict mode refuses to hide it: {exc}') from exc

    if bad_keys:
        summary['warnings'].append('nonfinite_values')
    if summary.get('first_vs_ref_mae') is not None and float(summary['first_vs_ref_mae']) > 1e-4:
        summary['warnings'].append('cache_sigma0_not_reference')
    summary['level'] = _rf_diagnostic_level(summary)
    return summary

def _rf_print_stability_summary(summary: Dict[str, Any]) -> None:
    level = summary.get('level', 'WARN')
    print(f'{_RF_PREFIX}   RF numerical sanity: {level}')
    print(
        f'{_RF_PREFIX}     cache_items={summary.get("cache_items")}  '
        f'sigmas={summary.get("sigmas")}  '
        f'first_sigma={_rf_scalar_fmt(summary.get("first_sigma"))}  '
        f'last_sigma={_rf_scalar_fmt(summary.get("last_sigma"))}  '
        f'finite_all={summary.get("finite_all")}'
    )
    print(
        f'{_RF_PREFIX}     std: ref={_rf_scalar_fmt(summary.get("ref_std"))}  '
        f'first={_rf_scalar_fmt(summary.get("first_std"))}  '
        f'final={_rf_scalar_fmt(summary.get("final_std"))}  '
        f'eps={_rf_scalar_fmt(summary.get("eps_std"))}  '
        f'final/eps={_rf_scalar_fmt(summary.get("final_eps_std_ratio"))}  '
        f'range=[{_rf_scalar_fmt(summary.get("std_min"))}, {_rf_scalar_fmt(summary.get("std_max"))}]'
    )
    print(
        f'{_RF_PREFIX}     final: mean={_rf_scalar_fmt(summary.get("final_mean"))}  '
        f'min={_rf_scalar_fmt(summary.get("final_min"))}  '
        f'max={_rf_scalar_fmt(summary.get("final_max"))}  '
        f'max_abs={_rf_scalar_fmt(summary.get("final_max_abs"))}  '
        f'max_abs_over_path={_rf_scalar_fmt(summary.get("max_abs_max"))}'
    )
    print(
        f'{_RF_PREFIX}     deltas: mean|Δz|={_rf_scalar_fmt(summary.get("dz_mean"))}  '
        f'max|Δz|={_rf_scalar_fmt(summary.get("dz_max"))}  '
        f'first_vs_ref_mae={_rf_scalar_fmt(summary.get("first_vs_ref_mae"))}  '
        f'final_vs_eps_mae={_rf_scalar_fmt(summary.get("final_vs_eps_mae"))}  '
        f'final_vs_eps_rmse={_rf_scalar_fmt(summary.get("final_vs_eps_rmse"))}  '
        f'final_vs_eps_cos={_rf_scalar_fmt(summary.get("final_vs_eps_cosine"))}'
    )
    warnings = summary.get('warnings') or []
    if warnings:
        print(f'{_RF_PREFIX}     warnings={warnings} bad_sigma_keys={summary.get("bad_sigma_keys", [])}')
    if level != 'PASS':
        print(
            f'{_RF_PREFIX}   ⚠ RF numerical sanity did not PASS. This is a diagnostic flag only; '
            f'it means inspect the printed stats and the generated image before trusting the run.'
        )
    else:
        print(
            f'{_RF_PREFIX}   RF numerical sanity PASS only means no obvious numeric collapse/divergence; '
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

def _normalize_sigma_float(value: Any) -> Optional[float]:
    if torch.is_tensor(value):
        v = float(value.detach().float().mean().item())
    else:
        v = float(value)
    if not math.isfinite(v):
        raise ValueError(f'Invalid sigma/timestep value {value!r}: not finite.')
    if 0.0 <= v <= 1.0:
        return max(0.0, min(1.0, v))
    if 1.0 < v <= 1000.0:
        return max(0.0, min(1.0, v / 1000.0))
    raise ValueError(f'Invalid sigma/timestep value {value!r}; expected [0,1] sigma or [1,1000] timestep.')

def _coerce_sigma_sequence(value: Any) -> Optional[List[float]]:
    """Convert a scheduler sigma/timestep list into normalized [0,1] floats.

    Strict mode: malformed schedules raise instead of returning None. A missing
    value still returns None so callers can distinguish "not provided" from
    "provided but broken".
    """
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
                raise ValueError(f'Invalid sigma sequence item {item!r}; expected tensor/int/float.')
    else:
        raise ValueError(f'Invalid sigma sequence type {type(value).__name__}; expected tensor/list/tuple.')

    out: List[float] = [_normalize_sigma_float(item) for item in flat]
    if len(out) < 2:
        raise ValueError(f'Invalid sigma sequence; expected at least 2 values, got {len(out)}.')
    dedup: List[float] = []
    for sigma in out:
        if not dedup or abs(dedup[-1] - sigma) > 1e-6:
            dedup.append(sigma)
    if len(dedup) < 2:
        raise ValueError(f'Invalid sigma sequence; deduplicated length is {len(dedup)}.')
    return dedup

def _rf_print_sampler_capture(verbose_flag: Any, found: Any, run_count: Any) -> None:
    if not _coerce_bool(verbose_flag):
        return
    print(
        f'{_RF_PREFIX} RFInversion sampler_sample: captured {len(found)} sigmas  '
        f'run={run_count}  seed=42'
    )


def _rf_print_build_requested(
    verbose_flag: Any,
    sampler_sigmas: Any,
    target_b: int,
    rf_cond_mode: str,
    cache_key: str,
    rf_ref_clean: Any,
    rf_kwargs: Any,
) -> None:
    if not _coerce_bool(verbose_flag):
        return
    print(f'{_RF_PREFIX}   RF build requested from RFInversion wrapper')
    print(f'{_RF_PREFIX}   {_rf_sequence_summary("sampler_sigmas", sampler_sigmas)}')
    print(f'{_RF_PREFIX}   target_b={target_b} cond_mode={rf_cond_mode} cache_key={str(cache_key)[:12]}')
    print(f'{_RF_PREFIX}   rf_ref_clean: {_rf_tensor_summary("rf_ref_clean", rf_ref_clean)}')
    print(f'{_RF_PREFIX}   rf_kwargs summary: {_rf_brief_obj(rf_kwargs)}')


def _rf_print_persistent_cache_hit(verbose_flag: Any, cache_key: str, built_cache: Any) -> None:
    if _coerce_bool(verbose_flag):
        print(f'{_RF_PREFIX}   RF persistent cache HIT key={str(cache_key)[:12]} cache_items={len(built_cache)}')


def _rf_print_persistent_cache_miss(verbose_flag: Any, cache_key: str) -> None:
    if _coerce_bool(verbose_flag):
        print(f'{_RF_PREFIX}   RF persistent cache MISS key={str(cache_key)[:12]} → building now')


def _rf_print_build_complete(
    verbose_flag: Any,
    built_cache: Any,
    sorted_sigmas: Any,
    eps: Any,
    rf_sanity: Dict[str, Any],
) -> None:
    if not _coerce_bool(verbose_flag):
        return
    print(
        f'{_RF_PREFIX}   RF build complete: cache_items={len(built_cache)} '
        f'built_sigmas={len(sorted_sigmas)} eps={_rf_tensor_summary("eps", eps)}'
    )
    _rf_print_stability_summary(rf_sanity)


def _rf_print_direct_fallback(verbose_flag: Any, sigma: float) -> None:
    raise RuntimeError(
        f'{_RF_PREFIX} No sampler sigmas captured; direct one-step RF path is disabled for σ={float(sigma):.6f}'
    )


def _rf_print_traceback(verbose_flag: Any, trace_text: str) -> None:
    if _coerce_bool(verbose_flag) and trace_text:
        print(trace_text)


def _rf_print_prepared(
    verbose_flag: Any,
    rf_mode: str,
    gamma: float,
    gamma_curve: float,
    norm_strength: float,
    pmi_alpha: float,
    model_info: Dict[str, Any],
) -> None:
    if not _coerce_bool(verbose_flag):
        return
    print(f'\n{_RF_PREFIX} ═══════════════════════════════════════')
    print(f'{_RF_PREFIX} RF INVERSION PREPARED')
    print(f'{_RF_PREFIX} ═══════════════════════════════════════')
    print(f'{_RF_PREFIX}   mode          : {rf_mode}')
    print(f'{_RF_PREFIX}   gamma         : {float(gamma):.4f}')
    print(f'{_RF_PREFIX}   gamma_curve   : {float(gamma_curve):.3f}')
    print(f'{_RF_PREFIX}   norm_strength : {float(norm_strength):.3f}')
    print(f'{_RF_PREFIX}   pmi_alpha     : {float(pmi_alpha):.3f}')
    print(f'{_RF_PREFIX}   seed          : 42 (internal fixed noise seed)')
    print(f'{_RF_PREFIX}   schedule      : captured from sampler at runtime; no SIGMAS input')
    print(f'{_RF_PREFIX}   output        : normal LATENT with RF metadata')
    print(f'{_RF_PREFIX}   wrapper       : standalone RF cache builder installed on MODEL')
    print(f'{_RF_PREFIX}   diagnostics   : verbose=True prints per-call/cache/conditioning details')
    _rf_print_model_identity(f'{_RF_PREFIX}   RFInversion', model_info)
    print(f'{_RF_PREFIX} ═══════════════════════════════════════\n')



def _untwist_format_scale_value(value: Any) -> str:
    """Format one scale value for compact debug logs without noisy trailing zeros."""
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f'Invalid RoPE scale value {value!r}: not finite.')
    text = f'{value_f:.6f}'.rstrip('0').rstrip('.')
    return text if text else '0'


def _untwist_scale_range(values: Any) -> str:
    """Return a compact [first ... last] summary from a 1D scale tensor/list."""
    if torch.is_tensor(values):
        if values.numel() <= 0:
            raise ValueError('Cannot summarize empty RoPE scale tensor.')
        flat = values.detach().float().flatten().to(device='cpu')
    else:
        flat = torch.as_tensor(values, dtype=torch.float32).flatten()
        if flat.numel() <= 0:
            raise ValueError('Cannot summarize empty RoPE scale values.')
    first = _untwist_format_scale_value(float(flat[0].item()))
    last = _untwist_format_scale_value(float(flat[-1].item()))
    return f'[{first} ... {last}]'


_AXIS0_ROPE_MODES = {'default', 'match_axes', 'constant'}

def _untwist_coerce_axis0_rope_mode(value: Any = None, legacy_scale: Any = None) -> str:
    """Mirror the main node's axis-0 RoPE mode normalization for debug output."""
    if value is None:
        if legacy_scale is not None:
            legacy_f = float(legacy_scale)
            return 'default' if legacy_f < 0.0 else 'constant'
        return 'default'

    mode = str(value or 'default').strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'legacy': 'default',
        'low': 'default',
        'low_scale': 'default',
        'match_axis1': 'match_axes',
        'match_axis_1': 'match_axes',
        'match_axis_1_plus': 'match_axes',
        'match_axes_1plus': 'match_axes',
        'same_as_axes': 'match_axes',
        'same_as_axis1': 'match_axes',
        'override': 'constant',
        'fixed': 'constant',
    }
    mode = aliases.get(mode, mode)
    if mode not in _AXIS0_ROPE_MODES:
        raise ValueError(f'Invalid axis0_rope_mode={value!r}; expected one of {sorted(_AXIS0_ROPE_MODES)}.')
    return mode

def _untwist_coerce_axis0_rope_scale(value: Any, default: float = 0.0) -> float:
    """Debug-side formatting clamp: axis0_rope_scale is non-negative now."""
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f'Invalid axis0_rope_scale={value!r}: not finite.')
    if value_f < 0.0:
        raise ValueError(f'Invalid axis0_rope_scale={value_f!r}; expected non-negative value.')
    return value_f


def _untwist_print_rope_scale_debug(
    stats: Optional[_RuntimeStats],
    cfg: Dict[str, Any],
    module_name: str,
    scale_vec: Any,
) -> None:
    """Print one compact RoPE scale-vector snapshot per UntwistingRoPE model call."""
    if not _coerce_bool(getattr(stats, 'verbose', False)):
        return
    if not isinstance(cfg, dict) or cfg.get('_rope_scale_debug_printed', False):
        return
    if not cfg.get('enabled', False) or int(cfg.get('cross_batch_target_batch', 0)) <= 0:
        return

    cfg['_rope_scale_debug_printed'] = True

    try:
        if not torch.is_tensor(scale_vec):
            scale_vec = torch.as_tensor(scale_vec)
        flat = scale_vec.detach().flatten()
        head_dim = int(flat.numel())
        axes_dims = [int(x) for x in (cfg.get('axes_dims') or [])]
        if not axes_dims or sum(axes_dims) != head_dim:
            axes_dims = [head_dim]

        if len(axes_dims) >= 2:
            axis0_dim = max(0, min(int(axes_dims[0]), head_dim))
            axis0 = flat[:axis0_dim]
            axis1_plus = flat[axis0_dim:]
        else:
            axis0 = flat
            axis1_plus = flat.new_empty((0,))

        progress = float(cfg.get('progress', 0.0))
        sigma = float(cfg.get('sigma', 0.0))
        low_scale = float(cfg.get('_debug_low_scale', 0.0))
        high_scale = float(cfg.get('_debug_high_scale', 0.0))
        axis0_rope_mode = _untwist_coerce_axis0_rope_mode(
            cfg.get('axis0_rope_mode', None),
            legacy_scale=cfg.get('axis0_rope_scale', None),
        )
        axis0_rope_scale = _untwist_coerce_axis0_rope_scale(
            cfg.get('axis0_rope_scale', 0.0), default=0.0
        )

        print(f'{_PREFIX}   progress={progress:.6f}  sigma={sigma:.6f}')
        print(f'{_PREFIX}   low_scale={low_scale:.6f}  high_scale={high_scale:.6f}')
        print(
            f'{_PREFIX}   axis0_rope_mode={axis0_rope_mode}  '
            f'axis0_rope_scale={axis0_rope_scale:.6f}'
        )
        print(f'{_PREFIX}   axis0={_untwist_scale_range(axis0)}')
        print(f'{_PREFIX}   axis1+={_untwist_scale_range(axis1_plus)}')
    except Exception as exc:
        raise RuntimeError(f'{_PREFIX} RoPE scale debug print failed; strict mode refuses to hide debug failure: {exc}') from exc


def _untwist_print_rope_scale_debug_from_cfg(
    stats: Optional[_RuntimeStats],
    cfg: Dict[str, Any],
    module_name: str,
    device: Any,
    dtype: Any,
    build_frequency_scale_vector: Any,
) -> None:
    """Strict RoPE debug print for adapter attention paths.

    The actual scale-vector builder stays in the main module and is passed in here
    so verbose_prints owns the formatting/printing without creating import cycles.
    """
    if not isinstance(cfg, dict) or cfg.get('_rope_scale_debug_printed', False):
        return
    if not callable(build_frequency_scale_vector):
        raise RuntimeError('RoPE scale debug requires callable build_frequency_scale_vector.')
    try:
        head_dim = int(cfg.get('head_dim', 0))
        if head_dim <= 0:
            raise RuntimeError(f'RoPE scale debug requires positive head_dim, got {head_dim}.')

        progress = float(cfg.get('progress', 0.0))
        high_start = float(cfg['high_scale_start'])
        high_end = float(cfg['high_scale_end'])
        low_start = float(cfg['low_scale_start'])
        low_end = float(cfg['low_scale_end'])
        high_scale = high_start + (high_end - high_start) * progress
        low_scale = low_start + (low_end - low_start) * progress
        beta = float(cfg.get('beta', 2.0))

        cfg['_debug_high_scale'] = float(high_scale)
        cfg['_debug_low_scale'] = float(low_scale)

        scale_vec = build_frequency_scale_vector(
            head_dim,
            cfg.get('axes_dims') or [],
            high_scale,
            low_scale,
            beta,
            device,
            dtype,
            runtime_cfg=cfg,
        )
        _untwist_print_rope_scale_debug(stats, cfg, module_name, scale_vec)
    except Exception as exc:
        raise RuntimeError(f'{_PREFIX} RoPE scale debug failed; strict mode refuses to hide failure: {exc}') from exc

def _untwist_print_patch_complete(stats: Optional[_RuntimeStats], rf_active: bool, adapter: Any) -> None:
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

__all__ = [
    '_PREFIX',
    '_RF_PREFIX',
    '_rf_prefix',
    '_coerce_bool',
    '_RuntimeStats',
    '_vprint',
    '_rf_vprint',
    '_rf_tensor_summary',
    '_rf_sequence_summary',
    '_rf_brief_obj',
    '_rf_tensor_stats',
    '_rf_scalar_fmt',
    '_rf_tensor_mae',
    '_rf_tensor_rmse',
    '_rf_tensor_cosine',
    '_rf_diagnostic_level',
    '_rf_stability_summary',
    '_rf_print_stability_summary',
    '_rf_model_identity',
    '_rf_print_model_identity',
    '_rf_step_iterator',
    '_rf_format_duration',
    '_rf_progress_snapshot',
    '_normalize_sigma_float',
    '_coerce_sigma_sequence',
    '_rf_print_sampler_capture',
    '_rf_print_build_requested',
    '_rf_print_persistent_cache_hit',
    '_rf_print_persistent_cache_miss',
    '_rf_print_build_complete',
    '_rf_print_direct_fallback',
    '_rf_print_traceback',
    '_rf_print_prepared',
    '_untwist_format_scale_value',
    '_untwist_scale_range',
    '_untwist_coerce_axis0_rope_mode',
    '_untwist_coerce_axis0_rope_scale',
    '_untwist_print_rope_scale_debug',
    '_untwist_print_rope_scale_debug_from_cfg',
    '_untwist_print_patch_complete',
]
