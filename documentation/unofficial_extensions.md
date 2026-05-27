# Unofficial Extensions

These options are experimental and are not part of the original Untwisting RoPE paper.

## `adain_on_v`

Extends AdaIN alignment from attention `Q/K` to also include `V`.

This can help ensure that the final image has a color scheme similar to that of the reference image.

## `post_attention_adain`

 Matches the target attention output statistics to the reference attention output.

This is borrowed from the [feature-injection idea in ConsiStory](https://arxiv.org/abs/2402.03286).

Unlike ConsiStory, this implementation does not use masks or spatial correspondence maps. It uses a simpler global AdaIN match.

## `axis0_rope_scale`

The paper recommends setting the RoPE's axis 0 to a value equal only to low_scale. However, as the timesteps increases, low_scale becomes a pretty high value, and for certain models not tested in the paper such as Z-image turbo, this introduces artifacts. That's why it's preferable to fix this value regardless of the denoising stage. Setting this to -1 restores the default behavior.
