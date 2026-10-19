# Common LoRA Training Failure Modes

Most failed LoRAs fail in one of a handful of predictable ways. Recognizing the symptom early saves hours of retraining.

## Overfitting

The classic failure: the adapter memorizes the training images instead of learning the concept. Symptoms include the subject appearing with the exact same background, pose, or clothing from the dataset no matter what the prompt says, plus plastic-looking skin and smeared textures. Causes are too many epochs for the dataset size, a learning rate that is too high, or a rank that is too large relative to the data. The fix is fewer epochs (often one to three for small datasets), a lower learning rate, and a smaller rank - then verify with test prompts that use backgrounds absent from the training set.

## Identity drift

The opposite problem: generations look like the subject only loosely. The model never locked onto distinguishing features because the dataset was too small, the trigger word appeared inconsistently in captions, or images of similar-but-different subjects were mixed in. Check the dataset first - identity drift is usually a data problem, not a hyperparameter problem.

## Prompt bleed

The adapter starts affecting unrelated generations: adding "skto woman" to a prompt about a cityscape changes the sky's color because one training image had an unusual sunset. This happens when captions under-described the scene, so the model attributed background details to the trigger word. Rewriting captions to explicitly describe backgrounds and lighting usually resolves it on retrain.

## Silent no-ops

A LoRA that does nothing at any strength is almost always a plumbing error: the adapter was trained for a different base model family (SD 1.5 weights loaded into an SDXL pipeline), the file was copied incompletely, or the strength slider sits at zero. Verify the base-model match and reload the file before suspecting the training itself.
