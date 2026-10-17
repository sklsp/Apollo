# ComfyUI Workflows

ComfyUI is a node-based interface for running diffusion models. Instead of fixed forms, you build a directed graph where each node performs one operation and passes tensors to its neighbors. The same graph can generate text-to-image, image-to-image, inpainting, or upscaling pipelines by rewiring nodes.

## Core nodes

Every workflow contains the same skeleton: a **CheckpointLoader** that loads the base model (and exposes its CLIP text encoder, VAE, and UNet as separate outputs), one or two **CLIPTextEncode** nodes for the positive and negative prompts, a **KSampler** that runs the denoising loop, and a **VAEDecode** plus **SaveImage** pair to turn latents into pixels. The KSampler's inputs — model, positive prompt, negative prompt, latent image, seed, steps, CFG, sampler name, and denoise strength — are the same settings described in the sampling documentation.

## Latent space plumbing

Nodes exchange latents rather than images wherever possible: an **EmptyLatentImage** node creates a blank canvas of chosen width and height for text-to-image work, while **VAEEncode** converts an uploaded image into latents for image-to-image or inpainting. The denoise strength on the KSampler controls how much of the source survives: 1.0 ignores it entirely (pure generation), values around 0.5 to 0.75 restyle while keeping composition, and low values near 0.3 only refine detail.

## Managing workflows

Workflows are saved as JSON files describing nodes and their connections, which makes them easy to version-control and share. Apollo stores its built-in workflow templates in a `workflows` directory and can load any compatible file from there; custom workflows added by the user appear in the picker automatically. Because the graph is data rather than code, swapping one node — for example replacing a plain upscaler with a tiled variant — changes behavior without touching anything else.
