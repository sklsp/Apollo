# Curating a LoRA Training Dataset

The quality of a LoRA is bounded by the quality of its training images. A well-curated small dataset beats a large messy one almost every time, so curation deserves more attention than hyperparameters.

## How many images

For a single subject - a person, a character, or an object - twenty to fifty images is the usual working range. Below about fifteen the model often fails to lock onto consistent identity details; above roughly eighty you are mostly paying in training time without gaining robustness, and you start needing more epochs control to avoid overfitting.

## Variety that matters

The dataset should vary what the subject does not: different angles (front, three-quarter, profile), lighting conditions, backgrounds, and distances from close-up to full body. What must stay consistent is the identity itself - face shape, hair color, distinctive features for a person; material and silhouette for an object. Including images where the subject looks like someone else actively teaches the model the wrong thing.

## Captions

Each image needs a caption that describes everything except the trigger word reserved for the subject. If the trigger is "skto woman", a good caption reads "a woman in a red coat walking through a market, overcast light" - describing scene and context while leaving identity to the adapter. Over-captioning (naming every trivial detail) makes the model attribute those details to the subject; under-captioning leaves it guessing.

## Cleaning steps

Before training: remove duplicates and near-duplicates (identical frames from a video count as one image), reject blurry or watermarked shots, and normalize resolution so nothing is far smaller than the training size - upscaling a 400-pixel thumbnail to 1024 only adds softness. Most curation pipelines report how many images were dropped at each stage so you can see where your dataset lost quality.
