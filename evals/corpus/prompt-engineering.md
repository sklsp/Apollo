# Prompt Engineering for Image Models

The text prompt is the main steering mechanism of a diffusion model, and small changes in wording can change the result dramatically. This document covers the conventions that reliably improve output quality.

## Structure: subject first

Models attend most strongly to the beginning of the prompt. Put the main subject in the first few words, then add setting, style, lighting, and finally technical modifiers such as "highly detailed" or "sharp focus". A prompt like "a weathered lighthouse on a cliff, storm clouds, dramatic lighting, highly detailed" will usually place the lighthouse front and center, while burying it at the end of a long sentence weakens its presence.

## Weighting with parentheses

Parentheses increase the relative importance of the words inside them: "(red fox):1.3" emphasizes "red fox", while "[red fox]:0.7" or double brackets reduce emphasis. Most interfaces accept explicit numeric weights in the form `(word:1.2)`. Weights between 1.1 and 1.5 are a safe range; pushing above about 2 often distorts anatomy or saturates colors because the model over-commits to that concept at every denoising step.

## Negative prompts

The negative prompt lists what should be avoided: "blurry, low quality, extra fingers, watermark". It works by conditioning the model to move away from those concepts during sampling. Keep it short — a wall of negative terms dilutes each one and can start fighting the positive prompt. For anatomy problems specifically, terms like "extra fingers", "deformed hands", and "bad anatomy" are the most effective entries.

## Style tokens

Certain phrases have become de facto style switches because they appear frequently in training data: "oil painting", "cyberpunk", "studio lighting", "35mm photograph", "isometric illustration". Combining two or three compatible style tokens works well; combining conflicting ones such as "watercolor" and "photorealistic" produces muddy, unpredictable results.

## What to avoid

All-caps text in prompts is treated the same as lowercase by most tokenizers, so it adds no emphasis. Extremely long prompts beyond roughly seventy-five tokens are truncated by the CLIP encoder, so anything past that point simply does not reach the model.
