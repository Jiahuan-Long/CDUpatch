# CDUPatch

## CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors

[English](README.md) | [简体中文](README.zh-CN.md)

This repository contains the code for the ACM Multimedia (ACM MM) paper:

**CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors**

CDUPatch is a research-oriented codebase for training and evaluating adversarial patches against YOLO-based object detectors. The project is built on top of the Ultralytics YOLOv3/YOLOv5 code structure and adds a custom adversarial patch pipeline for visible-light and mixed visible/infrared scenes.

## What This Repository Does

- Trains a universal adversarial patch with `train_patch.py`
- Applies the patch to detected targets with geometric transforms from `load_data.py`
- Evaluates attack quality with `test_patch.py`
- Keeps the `vis2infra/` module in the repository for visible-to-infrared patch mapping

In this codebase, the patch itself is the trainable object. The detector is used as a fixed target model, and optimization tries to reduce objectness and/or class confidence for selected categories.

## Key Files

- `train_patch.py`: main training script for adversarial patch optimization
- `test_patch.py`: evaluation script that measures precision, recall, and ASR
- `patch_config.py`: dataset paths, patch size, loss mode, and training settings
- `load_data.py`: patch transformer, patch applier, max-probability extractor, NPS, and TV utilities
- `vis2infra/RGB2infra_trainingwithtorch.py`: visible-to-infrared mapping helper
- `Readme_patch.md`: original local notes for patch evaluation setup

## Repository Layout

```text
CDUpatch/
├── train_patch.py
├── test_patch.py
├── patch_config.py
├── load_data.py
├── models/
├── utils/
├── vis2infra/
├── classify/
├── segment/
└── non_printability/
```

## Environment

- Python 3.8+
- PyTorch with CUDA is strongly recommended
- Install dependencies with:

```bash
pip install -r requirements.txt
```

## Before You Run

This project currently uses hard-coded local paths in several scripts. Update them before training or evaluation:

- `patch_config.py`
- `test_patch.py`
- dataset yaml files under `data/` if needed

The repository intentionally does not track large YOLO detector checkpoints such as `yolov3.pt`, `yolov5x.pt`, and similar `.pt` files. The visible-to-infrared mapping module under `vis2infra/` is kept in the repository, including its lightweight mapping assets.

## Release Status

- The current repository contains the main codebase for the paper.
- Related datasets and pretrained weights are being prepared and will be uploaded in a future update.
- Large YOLO detector checkpoints are still expected to be managed locally for now.

## Training

Use the main patch training entrypoint:

```bash
python train_patch.py
```

By default, the script:

- loads the detector weights configured in `patch_config.py`
- creates a random patch as the optimization target
- transforms and pastes the patch onto target boxes
- computes detector loss plus non-printability and total-variation penalties
- saves logs, images, and TensorBoard summaries under `patchTraining/`

## Evaluation

To test a saved patch on a validation set:

```bash
python test_patch.py
```

The evaluation script can:

- paste the patch onto detected targets
- save patched images and visualized detections
- export predicted labels
- compute precision, recall, and ASR

## Notes

- This is research code and assumes familiarity with adversarial patch experiments.
- The current implementation is CUDA-centric and may require small edits for CPU-only environments.
- Datasets, large YOLO model weights, generated outputs, and temporary experiment artifacts are excluded from version control.

## Acknowledgements

- Built on top of the Ultralytics YOLOv3/YOLOv5 project structure
- Customized for adversarial patch research on VisDrone and mixed visible/infrared data

## License

See `LICENSE` for licensing details inherited by this project and review upstream dependencies before redistribution.
