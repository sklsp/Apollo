# Apollo's LoRA Training Pipeline

Apollo automates the path from a folder of images to a trained LoRA adapter. The pipeline has four stages: dataset quality control, captioning, training configuration generation, and execution through an external trainer.

## Dataset quality control

Before anything is trained on, every image in the project folder passes automated checks. Images are decoded to verify they are not corrupt, near-duplicates are detected by perceptual hash and flagged for removal, and resolution is checked against a minimum threshold so tiny thumbnails do not silently degrade the dataset. The result is a quality report listing how many images passed, which were dropped, and why - the user reviews this before training starts.

## Captioning

Each surviving image receives a caption. If Ollama with a vision model is available locally, Apollo uses it to generate descriptive captions automatically; otherwise captions can be supplied manually as sidecar text files named after each image. Captions follow the curation guidance: describe the scene and context, keep the subject's identity out of the text so the adapter learns it from pixels alone.

## Configuration generation

Apollo writes a training configuration file in the YAML format expected by the Ostris AI Toolkit, an external local process that performs the actual GPU training. The generated YAML encodes the chosen base model, dataset path, trigger word, rank and alpha, learning rate, epochs, and batch size. Users can edit the generated file before launching; Apollo re-reads it at launch time so manual tweaks are respected.

## Execution and results

Training runs as a supervised background job: Apollo starts the toolkit process, polls its progress output, and records status transitions (queued, running, completed, failed) in its local job store. When training finishes, the resulting adapter file is copied into the LoRA library where it becomes selectable in generation workflows with an adjustable strength slider. Failed jobs keep their log files so the cause - usually out-of-memory or a path error - can be inspected without rerunning.
