# Stable Diffusion XL (SDXL)

Stable Diffusion XL, usually abbreviated SDXL, is a latent diffusion model released by Stability AI in 2023. It builds on the original Stable Diffusion architecture but with larger components and a higher native resolution, which makes it one of the most common base models for local image generation.

## Architecture changes over SD 1.5

SDXL keeps the overall pipeline — text encoder, U-Net denoiser, VAE decoder — but scales each part up. The UNet is roughly three times larger than in SD 1.5 and uses a deeper transformer-style attention block called the "big" attention layer. The variational autoencoder compresses images by a factor of eight per spatial dimension into a latent tensor with sixteen channels, compared to four channels in SD 1.5.

The text conditioning also changed: SDXL uses two CLIP encoders at once, an openai/clip-vit-large-patch14 model and an open_clip ViT-bigG model. The hidden states of both encoders are concatenated and fed into the UNet, which gives the model access to richer semantic information from the prompt than a single encoder can provide.

## Native resolution

SDXL was trained primarily at 1024 by 1024 pixels. Generating at that square size is where the model performs best; other aspect ratios work but should stay close to roughly one megapixel in total area. Pushing SDXL far outside its training range — for example a very wide banner or an extremely tall portrait — tends to produce duplicated subjects, warped anatomy, and degraded text rendering.

## Practical notes

Because of the larger UNet, SDXL needs more VRAM than SD 1.5: comfortable interactive generation starts around eight gigabytes with optimizations such as attention slicing enabled. Community fine-tunes and LoRAs trained on SDXL are not compatible with SD 1.5 checkpoints, so a workflow must commit to one base model family. Prompt style also shifted slightly: SDXL responds well to natural-language descriptions in addition to the comma-separated tag style popular for SD 1.5, though both still work.
