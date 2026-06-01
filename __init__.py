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
import latent_preview
from . import verbose_prints as vp
from .sdpa_fix import install_optimized_attention_override as _maybe_install_untwist_attention_override

_TRANSFORMER_CONFIG_KEY = model_adapters.CONFIG_KEY

# Module-level debug store. Comfy/KSampler may clone or pass model objects
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
    except Exception as exc:
        raise ValueError(f'Invalid gamma_curve value: {value!r}.') from exc
    if not math.isfinite(curve):
        raise ValueError(f'Invalid gamma_curve value: {value!r} is not finite.')
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
    except Exception as exc:
        raise ValueError(f'Invalid norm_strength value: {norm_strength!r}.') from exc
    if not math.isfinite(strength):
        raise ValueError(f'Invalid norm_strength value: {norm_strength!r} is not finite.')
    return max(0.0, min(1.0, strength))

def _coerce_strength01(value: Any, default: float = 0.0) -> float:
    try:
        strength = float(value)
    except Exception as exc:
        raise ValueError(f'Invalid strength value: {value!r}.') from exc
    if not math.isfinite(strength):
        raise ValueError(f'Invalid strength value: {value!r} is not finite.')
    return max(0.0, min(1.0, strength))

_AXIS0_ROPE_MODES = {'default', 'match_axes', 'constant'}

def _coerce_axis0_rope_mode(value: Any = None, legacy_scale: Any = None) -> str:
    """Normalize the axis-0 RoPE behavior selector."""
    if value is None:
        if legacy_scale is not None:
            try:
                return 'default' if float(legacy_scale) < 0.0 else 'constant'
            except Exception as exc:
                raise ValueError(f'Invalid legacy axis0_rope_scale value: {legacy_scale!r}.') from exc
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
        raise ValueError(f'Invalid axis0_rope_mode={value!r}. Expected one of {sorted(_AXIS0_ROPE_MODES)}.')
    return mode

def _coerce_axis0_rope_scale(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception as exc:
        raise ValueError(f'Invalid axis0_rope_scale value: {value!r}.') from exc
    if not math.isfinite(v):
        raise ValueError(f'Invalid axis0_rope_scale value: {value!r} is not finite.')
    return max(0.0, v)

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
    def __init__(self, pmi_dim: int = 22, eps: float = 1e-12) -> None:
        self.mean_velocity: Optional[torch.Tensor] = None
        self.prev_corrected_velocity: Optional[torch.Tensor] = None
        self.step_count: int = 0
        self.pmi_dim: int = max(1, int(pmi_dim))
        self.eps: float = float(eps)

    def reset(self) -> None:
        self.mean_velocity = None
        self.prev_corrected_velocity = None
        self.step_count = 0

    def _grad_norm(self, grad: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, grad.ndim))
        if not dims:
            return grad.detach().abs().clamp_min(self.eps)
        return torch.linalg.vector_norm(
            grad.detach().float(), ord=2, dim=dims, keepdim=True
        ).to(dtype=grad.dtype).clamp_min(self.eps)

    def update_and_correct(
        self,
        v_model: torch.Tensor,
        delta_t: float,
        t_next: float,
        strength: float = 1.0,
        post_update_corrected: bool = False,
    ) -> torch.Tensor:
        """
        Apply the PMI proximal-gradient velocity correction.

        strength is a radius multiplier: 0 disables the correction, 1 matches the
        official radius. Values between 0 and 1 are useful as a conservative UI
        control while preserving the official update form.
        """
        strength = max(0.0, min(1.0, float(strength)))
        if strength <= 0.0:
            return v_model

        dt = float(delta_t)
        if not math.isfinite(dt) or abs(dt) <= self.eps:
            return v_model

        t_next_f = float(t_next)
        if not math.isfinite(t_next_f):
            return v_model

        device = v_model.device
        dtype = v_model.dtype
        v_detached = v_model.detach()

        # Official PMI accumulates the time-weighted velocity, then normalizes it
        # by the next inverse-time value to get the mean-flow velocity.
        increment = (dt * v_detached).to(device=device, dtype=dtype)
        if self.mean_velocity is None:
            self.mean_velocity = increment.clone()
        else:
            self.mean_velocity = self.mean_velocity.to(device=device, dtype=dtype) + increment

        denom = t_next_f if abs(t_next_f) > self.eps else (self.eps if t_next_f >= 0.0 else -self.eps)
        pred_mean = (self.mean_velocity.to(device=device, dtype=dtype) / denom).detach()

        # ComfyUI samplers usually run under torch.no_grad(), and some setups use
        # inference-mode-style wrappers. Do not call autograd here. The official
        # PMI objective has a closed-form gradient:
        #   grad 0.5||v - v_mean||_2^2 = v - v_mean
        #   grad ||v - v_prev||_1       = sign(v - v_prev)
        # Computing it analytically keeps the official PMI update usable inside
        # Comfy's no-grad sampling path without adding any model evaluations.
        pred = v_detached
        grad = (pred.float() - pred_mean.float()).to(dtype=dtype)
        if self.prev_corrected_velocity is not None:
            prev = self.prev_corrected_velocity.to(device=device, dtype=dtype).detach()
            grad = grad + (pred.float() - prev.float()).sign().to(dtype=dtype)

        radius = math.sqrt(2.0 * self.pmi_dim + 3.0 * math.sqrt(2.0 * self.pmi_dim)) * abs(dt) * strength
        corrected = (pred - radius * (grad / self._grad_norm(grad))).to(dtype=dtype)

        self.prev_corrected_velocity = corrected.detach().clone()
        self.step_count += 1

        if post_update_corrected:
            self.mean_velocity = self.mean_velocity.to(device=device, dtype=dtype) + dt * corrected.detach()

        return corrected

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
    stats:          Optional[vp._RuntimeStats] = None,
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
        raise ValueError(
            f"Invalid rf_mode={mode!r}. Expected one of {sorted(valid_modes)}."
        )

    parameterization = getattr(stats, 'parameterization', 'unknown') if stats else 'unknown'

    device = ref_clean.device
    dtype  = ref_clean.dtype

    if eps is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        eps = torch.randn(ref_clean.shape, device=device, dtype=dtype, generator=rng)

    if sampler_sigmas is None:
        raise RuntimeError('RF trajectory build failed: sampler_sigmas is missing.')

    # Build sorted unique sigma grid starting from 0. Invalid entries are fatal.
    sigmas: List[float] = [0.0]
    for idx, s in enumerate(sampler_sigmas):
        try:
            sf = float(s)
        except Exception as exc:
            raise RuntimeError(
                f'RF trajectory build failed: sampler sigma at index {idx} is not numeric: {s!r}.'
            ) from exc
        if not math.isfinite(sf):
            raise RuntimeError(
                f'RF trajectory build failed: sampler sigma at index {idx} is not finite: {s!r}.'
            )
        sf = max(0.0, min(1.0, sf))
        if all(abs(sf - existing) > 1e-6 for existing in sigmas):
            sigmas.append(sf)
    sigmas = sorted(sigmas)
    if len(sigmas) <= 1:
        raise RuntimeError(
            'RF trajectory build failed: sampler sigma schedule did not contain any usable non-zero steps.'
        )

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

    vp._rf_vprint(stats,
        f'{vp._rf_prefix(stats)}   RF trajectory mode: {mode}  gamma={gamma:.4f}  '
        f'gamma_curve={gamma_curve:.3f}  '
        f'norm_strength={norm_strength:.3f}  '
        f'norm={"on" if norm_strength > 0.0 else "off"}  '
        f'parameterization={parameterization}\n'
        f'{vp._rf_prefix(stats)}   pmi_alpha={pmi_alpha_eff:.3f}  '
        f'PMI={"on" if use_pmi else "off"}'
    )

    # Print persistent RF inversion progress snapshots. This keeps every RF step
    rf_total_steps = max(1, len(sigmas) - 1)
    rf_progress_start_time = time.time()

    # RF step quality diagnostics.
    path_prev_speed: Optional[float] = None

    for step_index in vp._rf_step_iterator(rf_total_steps):
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
            nonlocal model_ok, vm_sum
            t_tensor = torch.full((z_in.shape[0],), sigma_val, device=device, dtype=dtype)
            with torch.no_grad():
                try:
                    raw = apply_model_fn(z_in, t_tensor, **base_model_kwargs)
                except Exception as exc:
                    raise RuntimeError(
                        f'RF trajectory build failed during model call at σ={sigma_val:.6f} '
                        f'mode={mode}{label}.'
                    ) from exc

            model_ok += 1
            v = _velocity_from_pred(z_in, raw, sigma_val, parameterization)
            vm_sum += float(v.abs().mean().item())
            return v, True, raw

        def _apply_pmi_if_enabled(
            v: torch.Tensor,
            pmi_time: float,
            *,
            post_update_corrected: bool = False,
        ) -> torch.Tensor:
            if not use_pmi:
                return v
            return pmi_state.update_and_correct(
                v,
                delta_t=delta,
                t_next=pmi_time,
                strength=pmi_alpha_eff,
                post_update_corrected=post_update_corrected,
            )

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

            v_mid_total = _apply_pmi_if_enabled(v_mid, sigma_cur, post_update_corrected=False)
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
                v_total = _apply_pmi_if_enabled(v_total, sigma_cur, post_update_corrected=True)
                z = z + delta * v_total
                _preview_once(step_i - 1, raw_preview_mid, z)
                extra = f'mid |v_model_mid|={vm_abs_mid:.5f}'
                if use_pmi:
                    extra += f'  PMI step={pmi_state.step_count}'
            else:
                v_total = gamma_eff * v_model + (1.0 - gamma_eff) * v_prior
                v_total = _apply_pmi_if_enabled(v_total, sigma_cur, post_update_corrected=True)
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

        vp._rf_vprint(stats,
            f'{vp._rf_prefix(stats)}     z_sigma step {step_i:02d}/{len(sigmas)-1}: '
            f'mode={mode}  γ_eff={gamma_eff:.4f}  '
            f'σ_prev={sigma_prev:.6f} -> σ={sigma_cur:.6f}  Δσ={delta:.6f}  '
            f'|model|={vm_abs:.5f}  |prior|={vp_abs:.5f}  |Δz|={dz_abs:.5f}  {extra}\n'
            f'{vp._rf_prefix(stats)}       z_σ mean={z_mean:.4f}  std={z_std:.4f}  '
            f'min={z_min:.4f}  max={z_max:.4f}'
        )
        path_prev_speed = vp._rf_print_step_quality(
            stats,
            ref_clean=ref_clean,
            eps=eps,
            z=z,
            sigma_cur=sigma_cur,
            delta=delta,
            dz_abs=dz_abs,
            path_prev_speed=path_prev_speed,
        )

        vp._rf_progress_snapshot(
            step_i,
            rf_total_steps,
            rf_progress_start_time,
            persistent=vp._coerce_bool(getattr(stats, 'rf_verbose', False)),
        )

    steps = max(1, len(sigmas) - 1)
    vp._rf_vprint(stats,
        f'{vp._rf_prefix(stats)}   RF schedule build: mode={mode}  sampler_sigmas={len(sampler_sigmas)}  '
        f'unique={len(sigmas)}  rf_steps={len(sigmas)-1}  '
        f'model_ok={model_ok}  failures={failures}\n'
        f'{vp._rf_prefix(stats)}     sigma_range=[{sigmas[0]:.6f}, {sigmas[-1]:.6f}]  '
        f'|model|={vm_sum/max(1, model_ok):.5f}  |prior|={vp_sum/steps:.5f}  '
        f'z_final std={z.std().item():.4f}  parameterization={parameterization}'
    )
    return cache, eps, sigmas

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
                seq = vp._coerce_sigma_sequence(obj.get(key))
                if seq is not None:
                    return seq
        for key, value in obj.items():
            key_l = str(key).lower()
            if any(word in key_l for word in ('sigma', 'timestep', 'schedule')):
                seq = vp._coerce_sigma_sequence(value)
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
        seq = vp._coerce_sigma_sequence(obj)
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

def _parse_active_blocks(blocks_str):
    active = set()
    if not isinstance(blocks_str, str) or not blocks_str.strip():
        return active
    for part in blocks_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                start, end = part.split('-', 1)
                start_i, end_i = int(start), int(end)
            except ValueError as exc:
                raise ValueError(f'Invalid active block range {part!r}.') from exc
            if end_i < start_i:
                raise ValueError(f'Invalid active block range {part!r}: end is smaller than start.')
            active.update(range(start_i, end_i + 1))
        else:
            try:
                active.add(int(part))
            except ValueError as exc:
                raise ValueError(f'Invalid active block index {part!r}.') from exc
    return active

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
        except Exception as exc:
            raise RuntimeError('Conditioning batch slice failed.') from exc
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
    if ref_conditioning is None:
        raise RuntimeError(
            'RF conditioning failed: ref_conditioning is required when RFInversion uses a reference latent.'
        )
    if target_b <= 0:
        raise RuntimeError(f'RF conditioning failed: invalid target batch size {target_b}.')

    try:
        merged, _forced = _merge_reference_conditioning_into_c(c, ref_conditioning, target_b)
        ref_only = _slice_conditioning_batch(merged, target_b, target_b * 2)
    except Exception as exc:
        raise RuntimeError('RF conditioning failed while merging reference conditioning.') from exc

    return _clone_conditioning_for_rf(ref_only), 'reference'

def _sigma_from_timestep(timestep: torch.Tensor) -> float:
    if not torch.is_tensor(timestep):
        raise RuntimeError('Sigma conversion failed: timestep is not a tensor.')
    try:
        val = float(timestep.detach().float().mean().item())
    except Exception as exc:
        raise RuntimeError('Sigma conversion failed: timestep could not be converted to float.') from exc
    if not math.isfinite(val):
        raise RuntimeError(f'Sigma conversion failed: timestep is not finite: {val!r}.')
    if 0.0 <= val <= 1.0:
        return max(0.0, min(1.0, val))
    if 1.0 < val <= 1000.0:
        return max(0.0, min(1.0, val / 1000.0))
    raise RuntimeError(f'Sigma conversion failed: unsupported timestep value {val!r}.')

def _sigma_to_progress(timestep: torch.Tensor, sampler_sigmas: List[float]) -> float:
    sigma = _sigma_from_timestep(timestep)

    if sampler_sigmas is None:
        raise RuntimeError('Progress conversion failed: sampler_sigmas is missing.')

    active: List[float] = []
    for idx, s in enumerate(sampler_sigmas):
        try:
            sf = float(s)
        except Exception as exc:
            raise RuntimeError(
                f'Progress conversion failed: sampler sigma at index {idx} is not numeric: {s!r}.'
            ) from exc
        if not math.isfinite(sf):
            raise RuntimeError(
                f'Progress conversion failed: sampler sigma at index {idx} is not finite: {s!r}.'
            )
        active.append(max(0.0, min(1.0, sf)))

    # The sampler schedule normally contains a terminal sigma=0 endpoint, but
    # the model is not evaluated there. Progress must span only real model calls
    # so *_end values are reached on the last denoising call.
    while active and active[-1] <= 1e-8:
        active.pop()

    if len(active) < 2:
        raise RuntimeError(
            'Progress conversion failed: sampler_sigmas did not contain at least two active model-call sigmas.'
        )

    idx = min(range(len(active)), key=lambda i: abs(active[i] - sigma))
    return max(0.0, min(1.0, idx / max(1, len(active) - 1)))

def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)

def _triangle_ramp01(progress: Any) -> float:
    """0→1 from progress 0→0.5, then 1→0 from progress 0.5→1."""
    p = _coerce_strength01(progress)
    return max(0.0, min(1.0, 1.0 - abs((2.0 * p) - 1.0)))

def _triangle_ramp_to_target(target: float, progress: Any) -> float:
    return _lerp(0.0, _coerce_strength01(target), _triangle_ramp01(progress))

def _repeat_conditioning_tree(obj: Any, src: int, tgt: int) -> Any:
    if torch.is_tensor(obj):
        try:
            if obj.ndim > 0 and int(obj.shape[0]) == src:
                return _repeat_to_batch(obj, tgt)
        except Exception as exc:
            raise RuntimeError('Conditioning tree repeat failed.') from exc
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
    except Exception as exc:
        raise RuntimeError(f'Could not convert reference metadata value to tensor: {value!r}.') from exc

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
        except Exception as exc:
            raise RuntimeError('Conditioning mask conversion failed for list/tuple source.') from exc
    try:
        return _num_tokens_to_valid_mask(int(source), batch, padded_tokens, device)
    except Exception as exc:
        raise RuntimeError(f'Conditioning mask conversion failed for source={source!r}.') from exc

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
            except Exception as exc:
                raise RuntimeError(f'Conditioning num-token merge failed for key={key!r}.') from exc
        if key in _MASK_CONDITIONING_KEYS:
            return forced_cap_mask
        return _repeat_conditioning_tree(value, target_b, target_b * 2)

    try:
        if value.ndim == 0 or int(value.shape[0]) != target_b:
            return value
    except Exception as exc:
        raise RuntimeError(f'Conditioning batch compatibility check failed for key={key!r}.') from exc

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
    head_dim, axes_dims, high_scale, low_scale, beta, device, dtype,
    axis0_rope_scale: Any = None,
    axis0_rope_mode: Any = None,
    runtime_cfg: Optional[Dict[str, Any]] = None,
):
    if not axes_dims or sum(int(x) for x in axes_dims) != head_dim:
        axes_dims = [head_dim]
    axes_dims = [int(x) for x in axes_dims]

    has_separate_axis0 = len(axes_dims) >= 2

    legacy_axis0_scale = None
    if isinstance(runtime_cfg, dict):
        legacy_axis0_scale = runtime_cfg.get('axis0_rope_scale', None)
        if axis0_rope_mode is None:
            axis0_rope_mode = runtime_cfg.get('axis0_rope_mode', None)
        if axis0_rope_scale is None:
            axis0_rope_scale = legacy_axis0_scale

    axis0_rope_mode = _coerce_axis0_rope_mode(
        axis0_rope_mode, legacy_scale=legacy_axis0_scale
    )
    axis0_rope_scale = _coerce_axis0_rope_scale(axis0_rope_scale, default=0.0)

    def _curve_scales(n_pairs: int) -> torch.Tensor:
        d_tilde = (
            torch.zeros(1, device=device, dtype=torch.float32)
            if n_pairs == 1
            else torch.linspace(0.0, 1.0, n_pairs, device=device, dtype=torch.float32)
        )
        return high_scale + (low_scale - high_scale) * d_tilde.pow(float(beta))

    pieces: List[torch.Tensor] = []
    for axis_idx, axis_dim in enumerate(axes_dims):
        n_pairs = axis_dim // 2
        if n_pairs <= 0:
            pieces.append(torch.ones(axis_dim, device=device, dtype=dtype))
            continue
        if has_separate_axis0 and axis_idx == 0:
            if axis0_rope_mode == 'match_axes':
                # Axis 0 uses the same per-pair curve as axes 1+.
                pair_scales = _curve_scales(n_pairs)
            elif axis0_rope_mode == 'constant':
                pair_scales = torch.full(
                    (n_pairs,), float(axis0_rope_scale), device=device, dtype=torch.float32
                )
            else:
                # Default/legacy behavior: axis 0 is flat at low_scale.
                pair_scales = torch.full(
                    (n_pairs,), float(low_scale), device=device, dtype=torch.float32
                )
        else:
            pair_scales = _curve_scales(n_pairs)
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

def _reference_variance_channel_mask(style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Build a [B, 1, H, D] mask from reference/style V variance across sequence.

    High mask values indicate channels whose reference activations vary strongly
    over tokens, which tends to correspond to texture/color/style-bearing
    features.
    """
    if not torch.is_tensor(style) or style.ndim != 4:
        raise RuntimeError(
            f'{vp._PREFIX} variance-gated V effects expected reference V as rank-4 BSHD, '
            f'got {tuple(style.shape) if torch.is_tensor(style) else type(style).__name__}.'
        )
    style_var = style.float().var(dim=1, keepdim=True, unbiased=False)
    style_var_max = style_var.amax(dim=-1, keepdim=True).clamp_min(eps)
    return (style_var / style_var_max).clamp(0.0, 1.0).detach()

def _cosine_gated_v_injection(
    v_t: torch.Tensor,
    v_r: torch.Tensor,
    strength: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Cosine-similarity-gated V injection.
    Only injects V_ref into V_target where they are semantically aligned
    (positive cosine similarity). Mismatched regions receive no injection.
    """
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return v_t

    v_tf = v_t.float()
    v_rf = v_r.float()

    t_u = v_tf / v_tf.norm(dim=-1, keepdim=True).clamp_min(eps)
    r_u = v_rf / v_rf.norm(dim=-1, keepdim=True).clamp_min(eps)

    # Per-token per-head cosine similarity -> gate in [0, 1].
    gate = (t_u * r_u).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)  # [B,S,H,1]

    # Gated delta: only aligned positions receive injection.
    delta = (v_rf - v_tf) * gate * strength
    return (v_tf + delta).to(dtype=v_t.dtype)

def _cross_batch_adain_qk(xq, xk, cfg, target_bsz, strength, eps=1e-6, xv=None):
    return_v = xv is not None
    if target_bsz <= 0 or xq.shape[0] < target_bsz * 2:
        return (xq, xk, xv) if return_v else (xq, xk)
    a = max(0.0, min(1.0, strength))
    if a <= 0.0:
        return (xq, xk, xv) if return_v else (xq, xk)
    seqlen = xq.shape[1]
    for s, e in (cfg.get('target_qk_adain_ranges') or []):
        s, e = max(0, int(s)), min(int(e), seqlen)
        if e <= s:
            continue
        q_t, k_t = xq[:target_bsz, s:e], xk[:target_bsz, s:e]
        q_r, k_r = xq[target_bsz:target_bsz*2, s:e], xk[target_bsz:target_bsz*2, s:e]
        xq[:target_bsz, s:e] = q_t * (1 - a) + _adain(q_t, q_r, eps) * a
        xk[:target_bsz, s:e] = k_t * (1 - a) + _adain(k_t, k_r, eps) * a
    return (xq, xk, xv) if return_v else (xq, xk)

# ═══════════════════════════════════════════════════════════════════════════════
# Shared Q/K/V effects
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_token_ranges(ranges: Any, seqlen: int) -> List[Tuple[int, int]]:
    """Clamp token ranges to the current sequence length and drop empty ranges."""
    out: List[Tuple[int, int]] = []
    for item in ranges or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        s = max(0, min(int(item[0]), int(seqlen)))
        e = max(s, min(int(item[1]), int(seqlen)))
        if e > s:
            out.append((s, e))
    return out

def _shared_effect_ranges(
    cfg: Dict[str, Any],
    seqlen: int,
    token_ranges: Any = None,
) -> List[Tuple[int, int]]:
    """Resolve explicit token ranges used by shared Q/K/V effects.
    """
    ranges = _normalize_token_ranges(token_ranges, seqlen)
    if ranges:
        return ranges
    raise RuntimeError(
        f'{vp._PREFIX} shared Q/K/V effects require explicit token_ranges; '
    )

def _apply_qkv_shared_effects(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: Dict[str, Any],
    target_bsz: int,
    module_name: str,
    *,
    layout: str = 'BSHD',
    token_ranges: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shared core Q/K/V effect stack for adapter-owned attention patches.

    Adapters expose where Q/K/V exist, their layout, and the applicable token
    range. This function owns shared user-facing Q/K/V features.

    Supported layouts:
      - BSHD: [batch, sequence, heads, head_dim]
      - BHSD: [batch, heads, sequence, head_dim]
    """
    if not isinstance(cfg, dict) or not cfg.get('enabled', False):
        return q, k, v
    if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v)):
        return q, k, v
    if int(target_bsz) <= 0 or int(v.shape[0]) < int(target_bsz) * 2:
        return q, k, v

    layout_u = str(layout or 'BSHD').upper()
    if layout_u == 'BSHD':
        q_bshd, k_bshd, v_bshd = q, k, v
        restore = lambda qq, kk, vv: (qq, kk, vv)
    elif layout_u == 'BHSD':
        q_bshd, k_bshd, v_bshd = q.movedim(1, 2), k.movedim(1, 2), v.movedim(1, 2)
        restore = lambda qq, kk, vv: (qq.movedim(1, 2), kk.movedim(1, 2), vv.movedim(1, 2))
    else:
        raise RuntimeError(
            f'{vp._PREFIX} shared QKV effects failed in {module_name}: '
            f'unsupported layout={layout!r}.'
        )

    if q_bshd.ndim != 4 or k_bshd.ndim != 4 or v_bshd.ndim != 4:
        raise RuntimeError(
            f'{vp._PREFIX} shared QKV effects failed in {module_name}: '
            f'expected Q/K/V as rank-4 after layout normalization, got '
            f'q={tuple(q_bshd.shape)}, k={tuple(k_bshd.shape)}, v={tuple(v_bshd.shape)}.'
        )

    seqlen = int(v_bshd.shape[1])
    ranges = _shared_effect_ranges(cfg, seqlen, token_ranges)

    # Shared pre-RoPE Q/K AdaIN.
    adain_strength = _coerce_strength01(cfg.get('adain_strength', 0.0)) if cfg.get('apply_adain') else 0.0
    if adain_strength > 0.0:
        cfg_for_adain = dict(cfg)
        cfg_for_adain['target_qk_adain_ranges'] = list(ranges)
        q_bshd = q_bshd.clone()
        k_bshd = k_bshd.clone()
        q_bshd, k_bshd = _cross_batch_adain_qk(
            q_bshd, k_bshd, cfg_for_adain, int(target_bsz), float(adain_strength)
        )
        cfg['_debug_qk_adain_strength'] = float(adain_strength)
        cfg['_debug_qk_adain_module'] = str(module_name)
        cfg['_debug_qk_adain_ranges'] = list(ranges)

    # Shared key-subspace Gram-Schmidt alignment.
    # Triangular ramp: 0.0 at progress=0, target at progress=0.5, back to 0.0 at progress=1.
    key_subspace_target = _coerce_strength01(cfg.get('key_subspace_alignment', 0.0))
    key_subspace_progress = _coerce_strength01(cfg.get('progress', 0.0))
    key_subspace_ramp = _triangle_ramp01(key_subspace_progress)
    key_subspace = _lerp(0.0, key_subspace_target, key_subspace_ramp)
    if key_subspace > 0.0:
        k_bshd = k_bshd.clone()
        for s, e in ranges:
            k_t = k_bshd[:target_bsz, s:e]
            k_r = k_bshd[target_bsz:target_bsz * 2, s:e]
            if k_t.shape != k_r.shape:
                raise RuntimeError(
                    f'{vp._PREFIX} shared key-subspace alignment failed in {module_name}: '
                    f'target/ref K range shape mismatch: target={tuple(k_t.shape)} ref={tuple(k_r.shape)}.'
                )

            k_tf = k_t.float()
            k_rf = k_r.float()
            dot_num = (k_tf * k_rf).sum(dim=-1, keepdim=True)
            dot_den = (k_rf * k_rf).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            proj = k_rf * (dot_num / dot_den)
            aligned = k_tf * (1.0 - key_subspace) + proj * key_subspace
            k_bshd[:target_bsz, s:e] = aligned.to(dtype=k_bshd.dtype)

        cfg['_debug_key_subspace_alignment_strength'] = float(key_subspace)
        cfg['_debug_key_subspace_alignment_target'] = float(key_subspace_target)
        cfg['_debug_key_subspace_alignment_progress'] = float(key_subspace_progress)
        cfg['_debug_key_subspace_alignment_ramp'] = float(key_subspace_ramp)
        cfg['_debug_key_subspace_alignment_module'] = str(module_name)
        cfg['_debug_key_subspace_alignment_ranges'] = list(ranges)

    # Shared cosine-gated V injection.
    # Triangular ramp: 0.0 at progress=0, target at progress=0.5, back to 0.0 at progress=1.
    cosine_v_inj_target = _coerce_strength01(cfg.get('cosine_gated_v_injection', 0.0))
    cosine_v_inj_progress = _coerce_strength01(cfg.get('progress', 0.0))
    cosine_v_inj_ramp = _triangle_ramp01(cosine_v_inj_progress)
    cosine_v_inj = _lerp(0.0, cosine_v_inj_target, cosine_v_inj_ramp)
    if cosine_v_inj > 0.0:
        v_bshd = v_bshd.clone()
        for s, e in ranges:
            v_t = v_bshd[:target_bsz, s:e]
            v_r = v_bshd[target_bsz:target_bsz * 2, s:e]
            if v_t.shape != v_r.shape:
                raise RuntimeError(
                    f'{vp._PREFIX} shared cosine-gated V injection failed in {module_name}: '
                    f'target/ref V range shape mismatch: target={tuple(v_t.shape)} ref={tuple(v_r.shape)}.'
                )

            v_bshd[:target_bsz, s:e] = _cosine_gated_v_injection(
                v_t, v_r, strength=cosine_v_inj
            )

        cfg['_debug_cosine_gated_v_injection_strength'] = float(cosine_v_inj)
        cfg['_debug_cosine_gated_v_injection_target'] = float(cosine_v_inj_target)
        cfg['_debug_cosine_gated_v_injection_progress'] = float(cosine_v_inj_progress)
        cfg['_debug_cosine_gated_v_injection_ramp'] = float(cosine_v_inj_ramp)
        cfg['_debug_cosine_gated_v_injection_module'] = str(module_name)
        cfg['_debug_cosine_gated_v_injection_ranges'] = list(ranges)

    # Shared variance-gated V-AdaIN.
    # Triangular ramp: 0.0 at progress=0, target at progress=0.5, back to 0.0 at progress=1.
    var_v_adain_target = _coerce_strength01(cfg.get('variance_gated_v_adain', 0.0))
    var_v_adain_progress = _coerce_strength01(cfg.get('progress', 0.0))
    var_v_adain_ramp = _triangle_ramp01(var_v_adain_progress)
    var_v_adain = _lerp(0.0, var_v_adain_target, var_v_adain_ramp)
    if var_v_adain > 0.0:
        v_bshd = v_bshd.clone()
        eps = 1e-6

        for s, e in ranges:
            v_t = v_bshd[:target_bsz, s:e]
            v_r = v_bshd[target_bsz:target_bsz * 2, s:e]
            if v_t.shape != v_r.shape:
                raise RuntimeError(
                    f'{vp._PREFIX} shared variance-gated V-AdaIN failed in {module_name}: '
                    f'target/ref V range shape mismatch: target={tuple(v_t.shape)} ref={tuple(v_r.shape)}.'
                )

            # [B, 1, H, D], normalized per head over the head-dim/channel axis.
            v_r_mask = _reference_variance_channel_mask(v_r, eps=eps)
            v_t_adain = _adain(v_t, v_r, eps=eps)
            alpha = (v_r_mask * var_v_adain).to(v_t.dtype)

            v_bshd[:target_bsz, s:e] = v_t * (1.0 - alpha) + v_t_adain * alpha

        cfg['_debug_variance_gated_v_adain'] = float(var_v_adain)
        cfg['_debug_variance_gated_v_adain_target'] = float(var_v_adain_target)
        cfg['_debug_variance_gated_v_adain_progress'] = float(var_v_adain_progress)
        cfg['_debug_variance_gated_v_adain_ramp'] = float(var_v_adain_ramp)
        cfg['_debug_variance_gated_v_adain_module'] = str(module_name)
        cfg['_debug_variance_gated_v_adain_ranges'] = list(ranges)

    q_bshd = _apply_implicit_attention_entropy_scaling(
        q_bshd, k_bshd,
        cfg,
        int(target_bsz),
        str(module_name),
        ranges,
    )

    return restore(q_bshd, k_bshd, v_bshd)

def _apply_attention_output_shared_effects(
    out_t: torch.Tensor,
    out_r: torch.Tensor,
    cfg: Dict[str, Any],
    target_bsz: int,
    module_name: str,
    *,
    layout: str = 'BSD',
    token_ranges: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shared post-attention output effect stack for adapter-owned attention patches.

    Adapters expose the target/reference attention outputs, layout, and optional
    architecture-specific token ranges.

    Supported layouts:
      - BSD/BSC: [batch, sequence, channels]
    """
    if not isinstance(cfg, dict) or not cfg.get('enabled', False):
        return out_t, out_r
    if not (torch.is_tensor(out_t) and torch.is_tensor(out_r)):
        return out_t, out_r
    if int(target_bsz) <= 0:
        return out_t, out_r

    layout_u = str(layout or 'BSD').upper()
    if layout_u not in ('BSD', 'BSC'):
        raise RuntimeError(
            f'{vp._PREFIX} shared attention-output effects failed in {module_name}: '
            f'unsupported layout={layout!r}.'
        )
    if out_t.ndim != 3 or out_r.ndim != 3:
        raise RuntimeError(
            f'{vp._PREFIX} shared attention-output effects failed in {module_name}: '
            f'expected target/ref outputs as rank-3, got target={tuple(out_t.shape)} ref={tuple(out_r.shape)}.'
        )
    if out_t.shape[0] != out_r.shape[0] or out_t.shape[2:] != out_r.shape[2:]:
        raise RuntimeError(
            f'{vp._PREFIX} shared attention-output effects failed in {module_name}: '
            f'target/ref output shape mismatch: target={tuple(out_t.shape)} ref={tuple(out_r.shape)}.'
        )

    seqlen = int(out_t.shape[1])
    ranges = _shared_effect_ranges(cfg, seqlen, token_ranges)

    # Triangular ramp: 0.0 at progress=0, target at progress=0.5, back to 0.0 at progress=1.
    post_a_target = _coerce_strength01(cfg.get('post_attention_adain_strength', 0.0))
    post_a_progress = _coerce_strength01(cfg.get('progress', 0.0))
    post_a_ramp = _triangle_ramp01(post_a_progress)
    post_a = _lerp(0.0, post_a_target, post_a_ramp)
    if post_a > 0.0:
        out_t = out_t.clone()
        for s, e in ranges:
            t_slice = out_t[:, s:e]
            r_slice = out_r[:, s:e]
            if t_slice.shape != r_slice.shape:
                raise RuntimeError(
                    f'{vp._PREFIX} shared post-attention AdaIN failed in {module_name}: '
                    f'target/ref range shape mismatch: target={tuple(t_slice.shape)} ref={tuple(r_slice.shape)}.'
                )
            out_t[:, s:e] = t_slice * (1.0 - post_a) + _adain(t_slice, r_slice, eps=1e-6) * post_a

        cfg['_debug_post_attention_adain_strength'] = float(post_a)
        cfg['_debug_post_attention_adain_target'] = float(post_a_target)
        cfg['_debug_post_attention_adain_progress'] = float(post_a_progress)
        cfg['_debug_post_attention_adain_ramp'] = float(post_a_ramp)
        cfg['_debug_post_attention_adain_module'] = str(module_name)
        cfg['_debug_post_attention_adain_ranges'] = list(ranges)

    return out_t, out_r

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
# Gram Attention-Entropy Scaling
# ═══════════════════════════════════════════════════════════════════════════════

def _expand_k_to_q_heads_for_logit_stats(k: torch.Tensor, q_heads: int) -> torch.Tensor:
    """Map KV heads to Q heads for grouped-query attention logit statistics."""
    if not torch.is_tensor(k) or k.ndim != 4:
        return k
    kv_heads = int(k.shape[2])
    q_heads = int(q_heads)
    if kv_heads == q_heads:
        return k
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise RuntimeError(
            f'{vp._PREFIX} Gram attention entropy scaling cannot expand KV heads: '
            f'q_heads={q_heads}, kv_heads={kv_heads}.'
        )
    return k.repeat_interleave(q_heads // kv_heads, dim=2)

def _exact_global_logit_variance_gram(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Uses:
      ||QKᵀ||²_F = Tr((QᵀQ)(KᵀK))

    It is not exact Shannon entropy; it is a VRAM-safe inverse-temperature
    control signal that preserves the downstream optimized attention backend.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise RuntimeError(
            f'{vp._PREFIX} Gram attention entropy scaling expected rank-4 Q/K, '
            f'got q={tuple(q.shape)}, k={tuple(k.shape)}.'
        )

    bq, s, q_heads, d = q.shape
    bk, t, k_heads, kd = k.shape
    if bq != bk or d != kd:
        raise RuntimeError(
            f'{vp._PREFIX} Gram attention entropy scaling Q/K shape mismatch: '
            f'q={tuple(q.shape)}, k={tuple(k.shape)}.'
        )
    if s <= 0 or t <= 0:
        return torch.ones((int(bq), int(q_heads)), device=q.device, dtype=torch.float32)

    k = _expand_k_to_q_heads_for_logit_stats(k, int(q_heads))
    if int(k.shape[2]) != int(q_heads):
        raise RuntimeError(
            f'{vp._PREFIX} Gram attention entropy scaling head mismatch after KV expansion: '
            f'q_heads={q_heads}, k_heads={int(k.shape[2])}.'
        )

    qf = q.float()
    kf = k.float()
    denom = float(max(1, int(s) * int(t)))

    # Mean of all logits per batch/head:
    #   E[QKᵀ] = dot(sum_i Q_i, sum_j K_j) / (S*T)
    sum_q = qf.sum(dim=1)  # [B,H,D]
    sum_k = kf.sum(dim=1)  # [B,H,D]
    mean = torch.einsum('bhd,bhd->bh', sum_q, sum_k) / denom

    # Second moment via Gram matrices:
    #   E[(QKᵀ)^2] = Tr((QᵀQ)(KᵀK)) / (S*T)
    gram_q = torch.einsum('bshd,bshe->bhde', qf, qf)  # [B,H,D,D]
    gram_k = torch.einsum('bthd,bthe->bhde', kf, kf)  # [B,H,D,D]
    second = torch.einsum('bhde,bhde->bh', gram_q, gram_k) / denom

    return (second - mean.pow(2)).clamp_min(float(eps))

def _apply_implicit_attention_entropy_scaling(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    cfg: Dict[str, Any],
    target_bsz: int,
    module_name: str,
    ranges: List[Tuple[int, int]],
) -> torch.Tensor:
    """
    Thermodynamic Attention-style inverse-temperature control without materializing
    B×S×T attention logits.

    This version uses the Gram identity to compute the exact global variance of
    the supplied pre-softmax QKᵀ logits per batch/head. Scaling target Q by
    sqrt(var_ref / var_target) makes the target's global logit variance match
    the reference while preserving optimized_attention_masked / FlashAttention.
    """
    if not isinstance(cfg, dict) or not cfg.get('enabled', False):
        return q_bshd
    entropy_scale = _coerce_strength01(cfg.get('attention_entropy_scaling', 0.0))
    if entropy_scale <= 0.0:
        return q_bshd
    if not (torch.is_tensor(q_bshd) and torch.is_tensor(k_bshd)):
        return q_bshd
    if q_bshd.ndim != 4 or k_bshd.ndim != 4:
        return q_bshd

    target_bsz = int(target_bsz)
    if target_bsz <= 0:
        return q_bshd
    if int(q_bshd.shape[0]) < target_bsz * 2 or int(k_bshd.shape[0]) < target_bsz * 2:
        return q_bshd

    try:
        scale_min = float(cfg.get('attention_entropy_scale_min', 0.35))
        scale_max = float(cfg.get('attention_entropy_scale_max', 3.0))
    except Exception as exc:
        raise ValueError('Invalid attention entropy scale clamp value.') from exc
    if not (math.isfinite(scale_min) and math.isfinite(scale_max)):
        raise ValueError('Invalid attention entropy scale clamp value: expected finite numbers.')
    scale_min = max(1e-3, min(scale_min, scale_max))
    scale_max = max(scale_min, scale_max)

    eps = 1e-6
    # Mutate the local Q tensor in place to avoid a full extra Q-sized clone.
    # Only the target image-token slices are changed.
    q_out = q_bshd
    applied_ranges: List[Tuple[int, int]] = []
    scale_means: List[float] = []
    var_t_means: List[float] = []
    var_r_means: List[float] = []

    q_seqlen = int(q_bshd.shape[1])
    k_seqlen = int(k_bshd.shape[1])

    for s, e in ranges or []:
        s_q = max(0, min(int(s), q_seqlen))
        e_q = max(s_q, min(int(e), q_seqlen))
        s_k = max(0, min(int(s), k_seqlen))
        e_k = max(s_k, min(int(e), k_seqlen))
        if e_q <= s_q or e_k <= s_k:
            continue

        q_t = q_bshd[:target_bsz, s_q:e_q]
        q_r = q_bshd[target_bsz:target_bsz * 2, s_q:e_q]
        k_t = k_bshd[:target_bsz, s_k:e_k]
        k_r = k_bshd[target_bsz:target_bsz * 2, s_k:e_k]

        if q_t.shape != q_r.shape:
            raise RuntimeError(
                f'{vp._PREFIX} Gram attention entropy scaling failed in {module_name}: '
                f'target/ref Q range shape mismatch: target={tuple(q_t.shape)} ref={tuple(q_r.shape)}.'
            )

        var_t = _exact_global_logit_variance_gram(q_t, k_t, eps=eps)
        var_r = _exact_global_logit_variance_gram(q_r, k_r, eps=eps)

        # Scaling Q by s scales logits by s and global logit variance by s².
        inv_temp = torch.sqrt(var_r / var_t.clamp_min(eps))
        inv_temp = inv_temp.clamp(scale_min, scale_max)
        inv_temp = 1.0 + entropy_scale * (inv_temp - 1.0)

        q_out[:target_bsz, s_q:e_q] = q_t * inv_temp.view(
            int(inv_temp.shape[0]), 1, int(inv_temp.shape[1]), 1
        ).to(dtype=q_bshd.dtype)

        applied_ranges.append((s_q, e_q))
        scale_means.append(float(inv_temp.detach().float().mean().cpu().item()))
        var_t_means.append(float(var_t.detach().float().mean().cpu().item()))
        var_r_means.append(float(var_r.detach().float().mean().cpu().item()))

    if applied_ranges:
        cfg['_debug_attention_entropy_scaling_strength'] = float(entropy_scale)
        cfg['_debug_attention_entropy_scaling_module'] = str(module_name)
        cfg['_debug_attention_entropy_scaling_ranges'] = list(applied_ranges)
        cfg['_debug_attention_entropy_method'] = 'gram_global_logit_variance'
        cfg['_debug_attention_entropy_inverse_temperature_mean'] = (
            sum(scale_means) / max(1, len(scale_means))
        )
        cfg['_debug_attention_entropy_target_variance_mean'] = (
            sum(var_t_means) / max(1, len(var_t_means))
        )
        cfg['_debug_attention_entropy_reference_variance_mean'] = (
            sum(var_r_means) / max(1, len(var_r_means))
        )
        cfg['_debug_attention_entropy_scale_min'] = float(scale_min)
        cfg['_debug_attention_entropy_scale_max'] = float(scale_max)

    return q_out

# ═══════════════════════════════════════════════════════════════════════════════
# Architecture detection
# ═══════════════════════════════════════════════════════════════════════════════

_ACTIVE_MODEL_ADAPTER: Any = None

# ═══════════════════════════════════════════════════════════════════════════════
# Context-refiner cap_mask patch
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_context_refiner_mask_modules(dm, stats):
    refiner = getattr(dm, 'context_refiner', None)
    if refiner is None:
        return 0, 0, 0
    if isinstance(refiner, (list, tuple, torch.nn.ModuleList)):
        modules = list(refiner)
    else:
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
                                except Exception as exc:
                                    raise RuntimeError(
                                        f'Context-refiner mask alignment failed: replacement={tuple(replacement_mask.shape)} '
                                        f'expected={tuple(orig_m.shape)}.'
                                    ) from exc
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

    vp._vprint(stats,
        f'{vp._PREFIX} Context-refiner mask patch: '
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
            except Exception as exc:
                raise RuntimeError('patchify_and_embed failed: rope_theta could not be read.') from exc

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
                except Exception as exc:
                    raise RuntimeError('patchify_and_embed failed: cap_size could not be converted to int.') from exc
                target_text_range = (0, cap0) if cap0 > 0 else None
                target_range = (max(0, cap0), int(img.shape[1]))

            real_range = target_range
            if target_range is not None:
                ts, te = int(target_range[0]), int(target_range[1])
                try:
                    real_tok   = (x.shape[-2] // p) * (x.shape[-1] // p)
                    real_range = (ts, min(ts + real_tok, te))
                except Exception as exc:
                    raise RuntimeError('patchify_and_embed failed: target real token range could not be computed.') from exc
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
        except Exception as exc:
            raise RuntimeError('patchify_and_embed strict metadata patch failed.') from exc

        return result

    dm.patchify_and_embed = types.MethodType(patched, dm)
    vp._vprint(stats, f'{vp._PREFIX} patchify_and_embed patched.')

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
    try:
        return latent_preview.prepare_callback(model_for_preview, total_steps)
    except Exception as exc:
        raise RuntimeError('RF preview callback creation failed.') from exc

def _rf_emit_preview(
    callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]],
    step: int,
    raw_pred: Optional[torch.Tensor],
    x_current: Optional[torch.Tensor],
    total_steps: int,
) -> None:
    """Emit one RF raw-pred preview frame."""
    if callback is None:
        raise RuntimeError('RF preview failed: callback is missing.')
    if not torch.is_tensor(raw_pred):
        raise RuntimeError('RF preview failed: raw_pred is not a tensor.')
    try:
        preview_latent = raw_pred[:1].detach()
        current = x_current[:1].detach() if torch.is_tensor(x_current) else preview_latent
        callback(int(step), preview_latent, current, int(total_steps))
    except Exception as exc:
        raise RuntimeError(f'RF preview frame failed at step {int(step) + 1}.') from exc

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
        'prefix': vp._PREFIX,
        'config_key': _TRANSFORMER_CONFIG_KEY,
        'lerp': _lerp,
        'cross_batch_adain_qk': _cross_batch_adain_qk,
        'build_frequency_scale_vector': _build_frequency_scale_vector,
        'apply_qkv_shared_effects': _apply_qkv_shared_effects,
        'apply_attention_output_shared_effects': _apply_attention_output_shared_effects,
        'print_rope_scale_debug': vp._untwist_print_rope_scale_debug,
        'print_rope_scale_debug_from_cfg': (
            lambda stats, cfg, module_name, device, dtype:
                vp._untwist_print_rope_scale_debug_from_cfg(
                    stats, cfg, module_name, device, dtype,
                    _build_frequency_scale_vector,
                )
        ),
        'patch_context_refiner_mask_modules': _patch_context_refiner_mask_modules,
        'patch_patchify_and_embed': _patch_patchify_and_embed,
    }

def _prepare_reference_conditioning_for_adapter(
    adapter: Any,
    ref_conditioning: Any,
    dm: Any,
    device,
    dtype,
    stats: Optional[vp._RuntimeStats] = None,
    label: str = '',
) -> Tuple[Any, str]:
    fn = getattr(adapter, 'prepare_reference_conditioning', None)
    if not callable(fn):
        raise RuntimeError(
            f'{vp._PREFIX} Adapter {type(adapter).__name__} does not implement '
            'prepare_reference_conditioning.'
        )
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
                'ref_conditioning': ('CONDITIONING',),
                'rf_mode': (['linear', 'rf_gamma', 'rf_gamma_rk2', 'fireflow'], {
                    'default': 'rf_gamma',
                    'tooltip': (
                        'Selects the ODE solver used to build the noisy reference trajectory: linear (no model calls -> random noise), rf_gamma (Euler), rf_gamma_rk2 (Runge-Kutta midpoint), or fireflow (FireFlow recurrence).'
                    ),
                }),
                'gamma': ('FLOAT', {
                    'default': 0.5,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Blends weight between model velocity and prior velocity (0 = pure prior / straight path, 1 = pure model); only used by rf_gamma and rf_gamma_rk2.'
                }),
                'gamma_curve': ('FLOAT', {
                    'default': 2.0,
                    'min': 0.0,
                    'max': 8.0,
                    'step': 0.05,
                    'tooltip': 'Applies a bell-shaped schedule to gamma across the sigma range, concentrating model influence toward mid-noise levels; 0 disables the curve.'
                }),
                'norm_strength': ('FLOAT', {
                    'default': 1.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.05,
                    'tooltip': "After each RF step, blends the latent's mean/std toward the linear target to prevent feature drift; 0 = off, 1 = full correction."
                }),
                'pmi_alpha': ('FLOAT', {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.05,
                    'tooltip': 'Proximal-Mean Inversion: 0 disables PMI; 1.0 matches the official radius. Applies to RF gamma, RK2, and FireFlow.'
                }),
                'verbose': ('BOOLEAN', {
                    'default': False,
                    'tooltip': 'Enable verbose logging.'
                }),
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
        pmi_alpha=0.0,
        verbose=False,
        ref_conditioning=None,
    ):
        rf_mode, gamma_curve = _normalize_rf_mode_and_gamma_curve(rf_mode, gamma_curve)
        norm_strength = _coerce_norm_strength(norm_strength)
        verbose_flag = vp._coerce_bool(verbose)

        if not isinstance(reference_latent, dict) or 'samples' not in reference_latent:
            raise RuntimeError("reference_latent must be a ComfyUI LATENT dict with 'samples'.")

        ref_clean = reference_latent['samples'].detach().clone()
        ref_clean = model.model.process_latent_in(ref_clean)

        # model_function_wrapper is passed ComfyUI model.apply_model, which returns
        # the denoised/x0-style prediction after model_sampling.calculate_denoised.
        # Keep the raw model type only as diagnostics; RF velocity conversion uses x0.
        model_info = vp._rf_model_identity(model)
        adapter = _select_model_adapter(model, model_info)
        detected_param = 'x0'
        dm_for_ref = None
        try:
            dm_for_ref = _safe_get_diffusion_model(model, adapter)
        except Exception as exc:
            raise RuntimeError(
                'RFInversion failed: could not access diffusion model for reference conditioning preprocessing.'
            ) from exc

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
            found = vp._coerce_sigma_sequence(sigmas)
            if found is not None:
                state['sampler_sigmas'] = found
                state['schedule_built'] = False
                state['schedule_sorted'] = None
                state['persistent_cache_key'] = None
                state['persistent_cache_hit'] = False
                state['cache'] = {0.0: ref_clean.detach().to(device='cpu').clone()}
                state['eps'] = None
                state['run_count'] = int(state.get('run_count', 0)) + 1
                state['preview_callback'] = _rf_make_preview_callback(model_clone, max(1, len(found) - 1))

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
                vp._rf_print_sampler_capture(verbose_flag, found, state["run_count"])
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
        rf_runtime_stats = vp._RuntimeStats(verbose=False, rf_verbose=verbose_flag)
        rf_runtime_stats.rf_prefix = vp._RF_PREFIX
        rf_runtime_stats.parameterization = detected_param

        def rf_model_function_wrapper(apply_model: Callable, args: Dict[str, Any]) -> torch.Tensor:
            state['wrapper_calls'] = int(state.get('wrapper_calls', 0)) + 1
            call_n = int(state['wrapper_calls'])
            debug_store['wrapper_calls'] = call_n

            input_x = args.get('input', None)
            timestep = args.get('timestep', None)
            c_in = args.get('c', {})
            c = c_in.copy() if isinstance(c_in, dict) else {}
            to = c.get('transformer_options', {}).copy()
            _maybe_install_untwist_attention_override(to)
            c['transformer_options'] = to
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

                    vp._rf_print_build_requested(
                        verbose_flag, sampler_sigmas, target_b, rf_cond_mode,
                        cache_key, rf_ref_clean, rf_kwargs,
                    )

                    cached_entry = _RF_PERSISTENT_TRAJECTORY_CACHE.get(cache_key)
                    if cached_entry is not None:
                        built_cache = _cache_to_device(cached_entry['cache'], input_x.device, input_x.dtype)
                        eps = cached_entry['eps'].to(device=input_x.device, dtype=input_x.dtype)
                        sorted_sigmas = list(cached_entry['built_sigmas'])
                        state['persistent_cache_hit'] = True
                        vp._rf_print_persistent_cache_hit(verbose_flag, cache_key, built_cache)
                    else:
                        state['persistent_cache_hit'] = False
                        preview_callback = state.get('preview_callback', None)
                        if preview_callback is None:
                            preview_callback = _rf_make_preview_callback(model_clone, max(1, len(list(sampler_sigmas)) - 1))
                            state['preview_callback'] = preview_callback
                        vp._rf_print_persistent_cache_miss(verbose_flag, cache_key)
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

                    rf_sanity = vp._rf_stability_summary(rf_ref_clean, eps, built_cache, list(sorted_sigmas))
                    state['last_stability_summary'] = rf_sanity
                    rf_latent['untwist_rf_stability_summary'] = rf_sanity
                    debug_store['stability_summary'] = rf_sanity

                    vp._rf_print_build_complete(verbose_flag, built_cache, sorted_sigmas, eps, rf_sanity)

                elif not state.get('schedule_built', False) and sampler_sigmas is None:
                    raise RuntimeError(
                        'RFInversion failed: sampler sigma schedule was not captured. '
                        'SAMPLER_SAMPLE did not run before the RF model wrapper was called.'
                    )

                cache = state.get('cache') if isinstance(state.get('cache'), dict) else {}
                cached = cache.get(sigma_key, None)
                cache_lookup = 'exact' if cached is not None else 'missing'
                if cached is None:
                    raise RuntimeError(
                        f'RFInversion failed: no exact RF cache entry for sigma={sigma_key:.6f}.'
                    )
                state['last_cache_lookup'] = cache_lookup
                debug_store['last_cache_lookup'] = cache_lookup


            except Exception as exc:
                state['last_error'] = repr(exc)
                debug_store['last_error'] = repr(exc)
                vp._rf_print_traceback(True, traceback.format_exc())
                raise RuntimeError('RFInversion standalone wrapper failed in.') from exc

            if old_model_function_wrapper is not None:
                return old_model_function_wrapper(apply_model, args)
            return apply_model(args['input'], args['timestep'], **args['c'])

        model_clone.set_model_unet_function_wrapper(rf_model_function_wrapper)

        vp._rf_print_prepared(
            verbose_flag, rf_mode, gamma, gamma_curve,
            norm_strength, pmi_alpha, model_info,
        )

        return (model_clone, rf_latent)

class UnofficialExtensions:
    CATEGORY = 'model_patches/Untwisting RoPE'
    RETURN_TYPES = ('UNTWISTING_ROPE_EXTENSIONS',)
    RETURN_NAMES = ('unofficial_extensions',)
    FUNCTION = 'build'
    DESCRIPTION = (
        'Optional unofficial toggles for Untwisting RoPE. '
        'These are intentionally separated from the main paper settings.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'post_attention_adain_strength': ('FLOAT', {
                    'default': 0.5,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Blend strength for matching the target attention output to the reference attention output.',
                }),
                'axis0_rope_mode': (['default', 'match_axes', 'constant'], {
                    'default': 'match_axes',
                    'tooltip': (
                        'Axis-0 RoPE behavior.\n'
                        'default -> Its values are equal to low_scale;\n'
                        'match_axes -> makes axis 0 use the same curve as axes 1+;\n'
                        'constant -> uses axis0_rope_scale.'
                    ),
                }),
                'axis0_rope_scale': ('FLOAT', {
                    'default': 1.0,
                    'min': 0.0,
                    'max': 8.0,
                    'step': 0.01,
                    'tooltip': 'RoPE scale used only when axis0_rope_mode = constant.',
                }),
                'cosine_gated_v_injection': ('FLOAT', {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Only aligned target/reference V tokens receive reference injection.',
                }),
                'attention_entropy_scaling': ('FLOAT', {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Matches target attention sharpness/diffuseness to the reference attention entropy.',
                }),
                'variance_gated_v_adain': ('FLOAT', {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': (
                        'Applies V AdaIN only on high-reference-variance channels. '
                    ),
                }),
                'key_subspace_alignment': ('FLOAT', {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': (
                        'Projects target keys onto reference keys to restrict routing '
                        'towards the reference key subspace.'
                    ),
                }),
            },
        }

    def build(
        self,
        post_attention_adain_strength: float = 0.0,
        axis0_rope_mode: str = 'default',
        axis0_rope_scale: float = 0.0,
        cosine_gated_v_injection: float = 0.0,
        attention_entropy_scaling: float = 0.0,
        variance_gated_v_adain: float = 0.0,
        key_subspace_alignment: float = 0.0,
    ):
        return ({
            'post_attention_adain_strength': _coerce_strength01(post_attention_adain_strength),
            'axis0_rope_mode': _coerce_axis0_rope_mode(axis0_rope_mode),
            'axis0_rope_scale': _coerce_axis0_rope_scale(axis0_rope_scale, default=0.0),
            'cosine_gated_v_injection': _coerce_strength01(cosine_gated_v_injection),
            'attention_entropy_scaling': _coerce_strength01(attention_entropy_scaling),
            'variance_gated_v_adain': _coerce_strength01(variance_gated_v_adain),
            'key_subspace_alignment': _coerce_strength01(key_subspace_alignment),
        },)

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
                'rf_inversion': ('LATENT',),
                'beta': ('FLOAT', {
                    'default': 50.0,
                    'min': 0.01,
                    'max': 100.0,
                    'step': 0.01,
                    'tooltip': 'Controls the steepness of the frequency scale curve. Higher values prevent the model from copying the reference image too closely.'
                }),
                'high_scale_start': ('FLOAT', {
                    'default': 1.0,
                    'min': -4.0,
                    'max': 8.0,
                    'step': 0.01,
                    'tooltip': 'Scale applied to high-frequency components. The higher the value, the more the final image will resemble the structure of the reference image.'
                }),
                'high_scale_end': ('FLOAT', {
                    'default': 0.00,
                    'min': -4.0,
                    'max': 8.0,
                    'step': 0.01,
                    'tooltip': 'Scale applied to high-frequency components. The higher the value, the more the final image will resemble the structure of the reference image.'
                }),
                'low_scale_start': ('FLOAT', {
                    'default': 1.0,
                    'min': -4.0,
                    'max': 8.0,
                    'step': 0.01,
                    'tooltip': 'Scale applied to low-frequency components. Controls the strength of the style image.'
                }),
                'low_scale_end': ('FLOAT', {
                    'default': 3.0,
                    'min': -4.0,
                    'max': 8.0,
                    'step': 0.01,
                    'tooltip': 'Scale applied to low-frequency components. Controls the strength of the style image.'
                }),
                'adain_strength': ('FLOAT', {
                    'default': 0.5,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'AdaIN aligns the target style statistics toward the reference.'
                }),
                'blocks': ('STRING', {
                    'default': '0-999',
                    'tooltip': 'Specify block ranges to patch, e.g -> 0-8, 28-37'
                }),
                'verbose': ('BOOLEAN', {
                    'default': False,
                    'tooltip': 'Enable verbose logging.'
                }),
            },
            'optional': {
                'unofficial_extensions': ('UNTWISTING_ROPE_EXTENSIONS',),
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
        blocks: str,
        adain_strength: float,
        verbose: bool = False,
        rf_inversion: Optional[Dict[str, Any]] = None,
        unofficial_extensions: Optional[Dict[str, Any]] = None,
    ):
        rf_active, rf_cfg, rf_state, ref_clean_cpu, ref_conditioning, rf_source = _rf_latent_get_config(rf_inversion)
        node_verbose = vp._coerce_bool(verbose)
        rf_verbose = vp._coerce_bool(rf_cfg.get('verbose', False))
        stats = vp._RuntimeStats(verbose=node_verbose, rf_verbose=rf_verbose)
        stats.rf_prefix = vp._RF_PREFIX
        debug_store = _rf_new_debug_store()

        rf_mode = str(rf_cfg.get('rf_mode', 'fireflow'))
        gamma = float(rf_cfg.get('gamma', 0.3))
        gamma_curve = float(rf_cfg.get('gamma_curve', 0.0))
        norm_strength = float(rf_cfg.get('norm_strength', 0.0))
        pmi_alpha = float(rf_cfg.get('pmi_alpha', 0.4))
        seed = int(rf_cfg.get('seed', 42))

        ext_cfg = unofficial_extensions if isinstance(unofficial_extensions, dict) else {}
        cosine_gated_v_injection = _coerce_strength01(ext_cfg.get('cosine_gated_v_injection', 0.0))
        variance_gated_v_adain = _coerce_strength01(ext_cfg.get('variance_gated_v_adain', 0.0))
        post_attention_adain_strength = _coerce_strength01(ext_cfg.get('post_attention_adain_strength', 0.0))
        axis0_rope_mode = _coerce_axis0_rope_mode(
            ext_cfg.get('axis0_rope_mode', None),
            legacy_scale=ext_cfg.get('axis0_rope_scale', None),
        )
        axis0_rope_scale = _coerce_axis0_rope_scale(ext_cfg.get('axis0_rope_scale', 0.0), default=0.0)
        attention_entropy_scaling = _coerce_strength01(ext_cfg.get('attention_entropy_scaling', 0.0))
        key_subspace_alignment = _coerce_strength01(ext_cfg.get('key_subspace_alignment', 0.0))

        if rf_active:
            stats.rf_sigma_cache = rf_state.get('cache', {}) if isinstance(rf_state.get('cache', {}), dict) else {}
            stats.rf_schedule_built = bool(rf_state.get('schedule_built', False))
            stats.parameterization = str(rf_inversion.get('untwist_rf_parameterization', 'unknown')) if isinstance(rf_inversion, dict) else 'unknown'
            debug_store['cache'] = stats.rf_sigma_cache
            debug_store['sampler_sigmas'] = list(rf_state.get('sampler_sigmas') or []) if isinstance(rf_state, dict) else []
            debug_store['built_sigmas'] = list(rf_state.get('schedule_sorted') or []) if isinstance(rf_state, dict) else []
            debug_store['parameterization'] = stats.parameterization

        vp._vprint(stats, f'\n{vp._PREFIX} ═══════════════════════════════════════')
        vp._vprint(stats, f'{vp._PREFIX} PATCH START  (split nodes: RFInversion + UntwistingRoPE)')
        vp._vprint(stats, f'{vp._PREFIX} ═══════════════════════════════════════')
        vp._vprint(stats, f'{vp._PREFIX} beta={beta}')
        vp._vprint(stats, f'{vp._PREFIX} high_scale: {high_scale_start:.3f} → {high_scale_end:.3f}')
        vp._vprint(stats, f'{vp._PREFIX} low_scale:  {low_scale_start:.3f} → {low_scale_end:.3f}')
        vp._vprint(stats,
            f'{vp._PREFIX} blocks: {blocks if blocks.strip() else "all"}  '
            f'adain={adain_strength:.2f}  '
            f'unofficial: '
            f'cosine_gated_v_injection={cosine_gated_v_injection:.2f}  '
            f'variance_gated_v_adain={variance_gated_v_adain:.2f}  '
            f'post_attention_adain_strength={post_attention_adain_strength:.2f}  '
            f'axis0_rope_mode={axis0_rope_mode}  '
            f'axis0_rope_scale={axis0_rope_scale:.3f}  '
            f'attention_entropy_scaling={attention_entropy_scaling:.2f}  '
            f'key_subspace_alignment={key_subspace_alignment:.2f}'
        )
        vp._vprint(stats, f'{vp._PREFIX} RF latent connected: {rf_active}  source={rf_source}')
        if rf_active:
            vp._vprint(stats,
                f'{vp._PREFIX} RF trajectory: mode={rf_mode}  gamma={gamma}  '
                f'gamma_curve={gamma_curve:.3f}  '
                f'norm_strength={norm_strength}  pmi_alpha={pmi_alpha}  '
                f'seed={seed}'
            )
            vp._vprint(stats, f'{vp._PREFIX} RF schedule: captured from sampler at runtime; no SIGMAS input')

        model_clone = model.clone()
        setattr(model_clone, '_untwisting_rope_rf_debug', debug_store)
        if rf_active:
            setattr(model_clone, '_untwisting_rope_rf_state', rf_state)
            setattr(model_clone, '_untwisting_rope_rf_config', rf_cfg)

        model_info = vp._rf_model_identity(model_clone)
        adapter = _select_model_adapter(model_clone, model_info)
        dm = _safe_get_diffusion_model(model_clone, adapter)
        vp._vprint(stats, f'{vp._PREFIX} Diffusion model type: {type(dm).__name__}')

        global _ACTIVE_MODEL_ADAPTER
        previous_adapter = _ACTIVE_MODEL_ADAPTER
        _ACTIVE_MODEL_ADAPTER = adapter
        try:
            patch_fn = getattr(adapter, 'patch_attention_modules', None)
            if not callable(patch_fn):
                raise RuntimeError(
                    f'{vp._PREFIX} Adapter {type(adapter).__name__} does not implement '
                    'patch_attention_modules.'
                )

            result = patch_fn(dm, stats, _adapter_helpers())
            if not isinstance(result, tuple):
                raise RuntimeError(
                    f'Unexpected adapter patch return from {type(adapter).__name__}: {result!r}'
                )
            if len(result) == 4:
                matched, installed, restored, patched_names = result
            elif len(result) == 3:
                matched, installed, restored = result
                patched_names = []
            else:
                raise RuntimeError(
                    f'Unexpected adapter patch return from {type(adapter).__name__}: {result!r}'
                )

            disp_name = getattr(adapter, 'DISPLAY_NAME', type(adapter).__name__)

            vp._vprint(stats, f'{vp._PREFIX} {disp_name} attention patch: matched={matched} installed={installed} restored={restored}')
            for n in patched_names:
                vp._vprint(stats, f'{vp._PREFIX}   - {n}')

            if installed == 0:
                raise RuntimeError(f'{vp._PREFIX} No {disp_name} attention blocks were patched.')
        finally:
            _ACTIVE_MODEL_ADAPTER = previous_adapter

        old_wrapper = model_clone.model_options.get('model_function_wrapper', None)

        parsed_blocks = _parse_active_blocks(blocks)

        def model_function_wrapper(apply_model: Callable, args: Dict[str, Any]) -> torch.Tensor:
            stats.wrapper_calls += 1
            call_n = stats.wrapper_calls

            input_x = args['input']
            timestep = args['timestep']
            c = args['c'].copy()
            cond_or_uncond = args.get('cond_or_uncond', None)
            to = c.get('transformer_options', {}).copy()
            _maybe_install_untwist_attention_override(to)

            sigma = _sigma_from_timestep(timestep)
            progress = _sigma_to_progress(timestep, list(rf_state['sampler_sigmas']))
            target_b = int(input_x.shape[0])

            cfg: Dict[str, Any] = {
                'enabled': True,
                'beta': float(beta),
                'high_scale_start': float(high_scale_start),
                'high_scale_end': float(high_scale_end),
                'low_scale_start': float(low_scale_start),
                'low_scale_end': float(low_scale_end),
                'active_blocks': parsed_blocks,
                'apply_adain': True,
                'adain_strength': float(adain_strength),
                'cosine_gated_v_injection': cosine_gated_v_injection,
                'variance_gated_v_adain': variance_gated_v_adain,
                'post_attention_adain_strength': post_attention_adain_strength,
                'axis0_rope_mode': axis0_rope_mode,
                'axis0_rope_scale': axis0_rope_scale,
                'attention_entropy_scaling': attention_entropy_scaling,
                'key_subspace_alignment': key_subspace_alignment,
                'cross_batch_target_batch': target_b if rf_active else 0,
                'progress': progress,
                'sigma': sigma,
                'wrapper_call': call_n,
            }
            default_cfg = getattr(adapter, 'default_runtime_cfg', None)
            if not callable(default_cfg):
                raise RuntimeError(
                    f'{vp._PREFIX} Adapter {type(adapter).__name__} does not implement '
                    'default_runtime_cfg.'
                )
            cfg.update(default_cfg(dm))
            to[_TRANSFORMER_CONFIG_KEY] = cfg

            input_for_model = input_x
            timestep_for_model = timestep
            ref_noisy = None
            sigma_key = round(float(sigma), 6)
            rf_cache_hit = False
            rf_cond_mode = 'not-connected'
            ref_mode = 'target-only'
            should_print = vp._coerce_bool(getattr(stats, 'verbose', False))

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
                            vp._rf_vprint(stats, f'{vp._rf_prefix(stats)} RFInversion persistent cache HIT: key={cache_key[:12]}  cache={len(built_cache)}')
                            ref_mode = 'RF sampler-sigma trajectory (persistent-cache hit)'
                            rf_state['persistent_cache_hit'] = True
                        else:
                            vp._rf_vprint(stats, f'{vp._rf_prefix(stats)} RFInversion persistent cache MISS: key={cache_key[:12]}  building trajectory')
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
                        raise RuntimeError(
                            'UntwistingRoPE failed: sampler sigma schedule was not captured and no RF trajectory was built. '
                            'SAMPLER_SAMPLE must run before UntwistingRoPE model calls.'
                        )

                    cache = rf_state.get('cache') if isinstance(rf_state.get('cache'), dict) else {}
                    cached = cache.get(sigma_key, None)
                    if cached is None:
                        raise RuntimeError(
                            f'UntwistingRoPE failed: no exact RF cache entry for sigma={sigma_key:.6f}.'
                        )
                    rf_cache_hit = True

                    ref_noisy = _repeat_to_batch(cached.to(device=input_x.device, dtype=input_x.dtype), target_b)

                    if ref_noisy.shape[-2:] == input_x.shape[-2:]:
                        input_for_model = torch.cat([input_x, ref_noisy], dim=0)
                        try:
                            if (torch.is_tensor(timestep)
                                    and timestep.ndim > 0
                                    and int(timestep.shape[0]) == target_b):
                                timestep_for_model = torch.cat([timestep, timestep], dim=0)
                            else:
                                timestep_for_model = _repeat_to_batch(timestep, target_b * 2)
                        except Exception as exc:
                            raise RuntimeError('UntwistingRoPE failed while duplicating timestep for reference batch.') from exc

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
                        except Exception as exc:
                            raise RuntimeError('UntwistingRoPE failed while duplicating cond_or_uncond metadata.') from exc
                    else:
                        raise RuntimeError(
                            f'UntwistingRoPE failed: spatial mismatch input_x={tuple(input_x.shape[-2:])} '
                            f'ref_noisy={tuple(ref_noisy.shape[-2:])}. '
                            f'Make sure the resolution of the reference image fed into the RF inversion node matches the final image resolution (same width and height).'
                        )
                except RuntimeError:
                    raise
                except Exception as exc:
                    raise RuntimeError('UntwistingRoPE RF latent preparation failed.') from exc

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

            if should_print:
                vp._untwist_print_rope_scale_debug_from_cfg(
                    stats, cfg, 'model_wrapper', input_x.device, input_x.dtype,
                    _build_frequency_scale_vector,
                )

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
                    raise RuntimeError(
                        f'UntwistingRoPE failed while caching RF debug latents at σ={float(sigma):.6f}.'
                    ) from exc

                return target_pred

            return raw_result

        model_clone.model_options = _clone_model_options(model_clone.model_options)
        model_clone.set_model_unet_function_wrapper(model_function_wrapper)

        vp._untwist_print_patch_complete(stats, rf_active, adapter)

        return (model_clone,)

NODE_CLASS_MAPPINGS = {
    'RFInversion': RFInversion,
    'UnofficialExtensions': UnofficialExtensions,
    'UntwistingRoPE': UntwistingRoPE,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'RFInversion': 'RF Inversion',
    'UnofficialExtensions': 'Unofficial Extensions',
    'UntwistingRoPE': 'Untwisting RoPE',
}
