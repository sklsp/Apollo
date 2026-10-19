# Sampling Methods and CFG Scale

Two settings control how a diffusion model turns noise into an image: the sampler (which numerical method steps through the denoising process) and the classifier-free guidance scale, usually written as CFG.

## What CFG does

Classifier-free guidance interpolates between "what the model thinks is likely" and "what matches the prompt". A CFG of 1 means pure model output with no extra steering; a CFG of 7 pulls each denoising step strongly toward the prompt's concepts. Low values (under about 4) give dreamy, loosely related images; high values (above about 9) make colors oversaturated and can introduce banding or harsh edges because every step is over-corrected toward the text. For most SDXL work a CFG between 5 and 7 is a sensible starting point.

## Common samplers

- **Euler**: simple and fast, good quality at moderate step counts; a solid default when speed matters.
- **DPM++ 2M Karras**: the most popular high-quality sampler; it pairs a second-order prediction with a Karras noise schedule that front-loads detail work in early steps. It typically needs 20 to 30 steps.
- **DDIM**: deterministic for a given seed, which makes it useful when reproducibility matters more than peak quality.
- **UniPC**: a newer fast sampler that reaches good quality with fewer steps than DPM++ variants.

## Steps and seeds

The step count sets the budget of denoising passes: 20 is usually enough for DPM++ 2M Karras, while Euler may want 30 to 40 for fine detail. The seed fixes the initial random noise; keeping the same seed with identical settings reproduces the same image, which is essential when comparing prompts or sampler changes one variable at a time.

## Choosing a starting point

A practical default combination is DPM++ 2M Karras, 25 steps, CFG 6, at the model's native resolution. Change only one setting at a time: if an image looks noisy, raise steps before raising CFG; if it ignores the prompt, raise CFG before rewriting the prompt.
