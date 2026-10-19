# Resolution and Upscaling

Diffusion models generate at a fixed native resolution; getting larger or sharper images requires a second stage. This document covers how the two stages fit together.

## Why not just generate big?

Generating directly at 4K would require a latent canvas far outside the model's training distribution: subjects duplicate, anatomy breaks down, and memory usage grows with pixel count. The standard approach is to generate at native resolution - 1024 by 1024 for SDXL - and then upscale the result with a dedicated upscaler network.

## Upscaler networks

Modern upscalers are small convolutional networks trained on paired low/high-resolution crops; they learn plausible high-frequency detail rather than merely interpolating pixels. Popular choices include ESRGAN variants and SwinIR-style transformers. An upscaler multiplies the image by a fixed factor, commonly 2x or 4x, so a 1024-pixel source becomes 2048 or 4096 pixels after one pass.

## Tiled upscaling

Upscaling a full 4K canvas in one go can exceed GPU memory, so upscalers run in tiles: the image is split into overlapping patches (typically 512 by 512 with some overlap), each patch is upscaled independently, and the results are blended back together. The overlap prevents visible seams at tile boundaries; too little overlap shows faint grid lines, while excessive overlap wastes compute without improving quality.

## Two-stage refinement

A common high-quality recipe is: generate at native resolution, upscale 2x with a strong upscaler, then run a light diffusion pass over the upscaled image (image-to-image with low denoise strength around 0.3 to 0.4). The second stage lets the diffusion model re-render fine detail - skin texture, fabric weave, foliage - at the higher resolution instead of trusting the upscaler's hallucinated detail alone. This is slower but usually the best-looking result for hero images.
