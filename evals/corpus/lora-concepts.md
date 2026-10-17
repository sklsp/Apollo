# LoRA: Low-Rank Adaptation

LoRA (Low-Rank Adaptation) is a technique for fine-tuning a large model cheaply. Instead of updating all the weights of a diffusion UNet, which would require copying hundreds of gigabytes and days of GPU time, LoRA learns a small set of additional matrices that are added to the original weights at inference time.

## How it works

For a weight matrix W in the network, LoRA freezes W and learns two smaller matrices A and B such that the update is approximated as W + A·B. The product A·B has the same shape as W but far fewer parameters, because A and B are "low rank": their inner dimension, called the rank, is much smaller than the matrix width. During training only A and B are updated; during generation the combined weights are applied on the fly with negligible overhead.

## Rank and alpha

The **rank** controls how much information the adapter can carry. Low ranks (4 to 8) capture a single strong concept such as one person's face or one art style, and they resist overfitting. Higher ranks (32 and up) can encode more complex behavior but need more data and are easier to over-train.

**Alpha** is a scaling factor applied to the learned update; the effective strength of the adapter at inference time is alpha divided by rank. Keeping alpha equal to rank gives a neutral starting point, and most training tools expose both values so users can tune relative strength without retraining.

## Why LoRA instead of full fine-tuning

A LoRA for SDXL typically occupies 10 to 200 megabytes on disk versus several gigabytes for a full checkpoint. It trains in minutes to an hour on a single consumer GPU, and multiple LoRAs can be stacked on the same base model at generation time with independent strength sliders — something impossible when each variant is a separate full checkpoint.
