# Unofficial Extensions

These options are experimental and are not part of the original Untwisting RoPE paper.

## `adain_on_v`

Extends AdaIN alignment from attention `Q/K` to also include `V`.

This can help ensure that the final image has a color scheme similar to that of the reference image.

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

## `orthogonal_v_injection`

Uses Gram-Schmidt projection to inject only the orthogonal component of the reference V tensor to the target V tensor, transferring texture/color/style while reducing semantic bleed.

Gives pretty good results at low strength (~0.2).
