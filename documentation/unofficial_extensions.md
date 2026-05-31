# Unofficial Extensions

These options are experimental and are not part of the original Untwisting RoPE paper.

## `post_attention_adain`

 Matches the target attention output statistics to the reference attention output.

This is borrowed from the [feature-injection idea in ConsiStory](https://arxiv.org/abs/2402.03286).

Unlike ConsiStory, this implementation does not use masks or spatial correspondence maps. It uses a simpler global AdaIN match.

## `axis0_rope_mode`

The paper recommends setting the RoPE's axis 0 to a value equal to `low_scale` (uniform across all frequencies) for the only model they tested which was flux.1-dev. Perhaps this method works very well for that specific model, but for other models such as Z-Image Turbo, the result can be disastrous. It ends up amplifying the signal too much.

<img width="720" alt="combined_image" src="https://github.com/user-attachments/assets/21fd928d-6e8e-4827-8095-40fa534de95d" />


You have three choices:
- `default` -> As the paper intended
- `match_axes` -> axis0 ends up behaving exactly like the other axes (best results).
- `constant` -> You set up your own `axis0_rope_scale` value 

## `cosine_gated_v_injection`

Injects reference V into target V only where their cosine similarity is positive.

This makes V injection less aggressive than a plain blend and reduces artifacts from pushing reference features into unrelated target regions.

Conceptually inspired by [CACTIF's similarity-filtered attention](https://arxiv.org/abs/2505.16360), but implemented here as a lightweight token-local V-space gate.

## `attention_entropy_scaling`

Scales the target's variance of the attention-score `QKᵀ` towards the reference's. It does not compute the full Shannon entropy and uses a Gram-matrix approximation as a cheaper entropy-like signal.

Can work well at full strength for making images look cleaner.

## `variance_gated_v_adain`

Applies AdaIN to the target V tensor but only on reference channels with high variance.

This makes the image even cleaner and further enhances the transfer style.

## `key_subspace_alignment`

Projects the target K tensor onto the reference K direction.

It's really effective at intensifying style transfer at low strength values (~0.1). 
