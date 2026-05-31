# ComfyUI-Untwisting-RoPE: Parameters section

## RF Inversion Node

RF Inversion builds a noisy trajectory on the reference image, so the model sees a properly noise-matched version of the reference image at every denoising step.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reference_latent` | — | The clean reference image latent that the inversion trajectory is built from. |
| `ref_conditioning` | *(optional)* | Text conditioning associated with the reference image. In practice, it's better to put the target conditioning to it. |
| `rf_mode` | `gnri` | Selects the ODE solver used to build the noisy reference trajectory: `linear` (no model calls -> random noise), `rf_gamma` (Euler), `rf_gamma_rk2` (Runge-Kutta midpoint), `fireflow` [(FireFlow recurrence)](https://arxiv.org/abs/2412.07517?utm_source=chatgpt.com) or `gnri` [(Guided Newton-Raphson Inversion)](https://barakmam.github.io/rnri.github.io/) |
| `gamma` | `0.50` | Blends weight between model velocity and prior velocity (0 = pure prior / straight path, 1 = pure model); only used by `rf_gamma` and `rf_gamma_rk2`. |
| `gamma_curve` | `2.00` | Applies a bell-shaped schedule to `gamma` across the sigma range, concentrating model influence toward mid-noise levels; 0 disables the curve. |
| `norm_strength` | `1.00` | After each RF step, blends the latent's mean/std towards the linear target to prevent feature drift; 0 = off, 1 = full correction. |
| `pmi_alpha` | `0.00` | [PMI (Proximal-Mean Inversion)](https://arxiv.org/abs/2602.11850) smooths out the velocity estimation by using a running mean across steps, 0 disables PMI. |

---

## Untwisting RoPE Node

Untwisting RoPE patches the model's attention layers to let the target image attend to the reference image's keys and values (KV) and then rescales RoPE frequencies to enable style transfer without any training.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rf_inversion` | *(optional)* | Fetches the inverted noisy latents created by the `RF Inversion` node. |
| `beta` | `50.00` | Controls the steepness of the frequency scale curve. Higher values prevent the model from copying the reference image too closely. |
| `high_scale` | `1.05`  | Scale applied to high-frequency components. The higher the value, the more the final image will resemble the structure of the reference image.|
| `low_scale` | `3.0` | Scale applied to low-frequency components. Basically controls the strength of the style transfer. |
| `adain_strength` | `0.50` | [AdaIN (Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization)](https://arxiv.org/abs/1703.06868) aligns the target style statistics toward the reference. |
| `blocks` | `0-999` | Block indices to which the reference attention patch is applied. |
