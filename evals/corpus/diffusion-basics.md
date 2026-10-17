# How Diffusion Models Work

A diffusion model is a generative model that learns to produce images by reversing a gradual noising process. Instead of predicting an entire image in one step, the model learns to remove a little noise at a time until a coherent picture emerges from random static.

## The forward process

Training starts with real images. The model is shown an image and then asked to predict the noise that was added to it. Concretely, a copy of the training image is corrupted by adding Gaussian noise in many small steps: first almost no noise, then slightly more, until the image is indistinguishable from pure random noise. This sequence of increasingly noisy versions is called the diffusion schedule.

## The reverse process

The network itself is usually a U-Net that takes a noisy image and a step counter as input and outputs an estimate of the noise present in that image. Subtracting the predicted noise from the noisy image yields a slightly cleaner version. Repeating this denoising loop from pure noise produces the final generated image. Each individual pass through the network is called one sampling step, and typical generations use between 20 and 50 steps.

## Latent diffusion

Running the full noising process on raw pixels is expensive, so modern systems such as Stable Diffusion work in a compressed latent space instead. An autoencoder first compresses an image into a small tensor of latent values, roughly eight times smaller in each spatial dimension. The diffusion process then runs entirely on these latents, and only at the very end does the decoder turn the final clean latent back into pixels. This is why the same model can generate images at different resolutions without retraining.

## Conditioning with text

To make generation follow a prompt, the noise prediction network is conditioned on an embedding of the text prompt produced by a language encoder such as CLIP. The encoder turns the prompt into a sequence of vector tokens, and cross-attention layers let every part of the image attend to those tokens. This conditioning is what makes the model produce "a red fox in snow" rather than random texture: the text embedding steers which denoising path the latent follows at every step.

## Why steps matter

Fewer sampling steps are faster but give the model less opportunity to refine structure, so very short runs can look blurry or contain artifacts. More steps cost more compute and usually improve fine detail up to a point of diminishing returns. The exact trade-off depends on the sampler used, which is covered in the sampling documentation.
