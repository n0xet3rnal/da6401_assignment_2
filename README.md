# DA6401 Assignment 2: Multi-Task Pet Perception

This project implements a unified computer vision pipeline on the Oxford-IIIT Pet dataset with four capabilities:

- Task 1: Breed classification (37 classes)
- Task 2: Pet localization (bounding box regression)
- Task 3: Pet segmentation (3-class trimap segmentation)
- Task 4: Unified multitask inference (classification + localization + segmentation)

The models are based on a VGG11 backbone with optional BatchNorm and custom dropout.

## Directory Structure

```text
da6401_assignment_2/
|-- .gitignore                     # Ignore rules used for generated/local artifacts
|-- README.md
|-- inference.py                   # Placeholder for generic inference/evaluation
|-- requirements.txt               # Python dependencies
|-- train.py                       # Training entrypoint for Tasks 1/2/3
|
|-- checkpoints/
|   |-- checkpoints.md             # Checkpoint naming and submission notes
|
|-- data/
|   |-- pets_dataset.py            # Dataset + transforms + train/val split logic
|
|-- losses/
|   |-- __init__.py
|   |-- iou_loss.py                # Custom IoU loss for localization
|
|-- models/
|   |-- __init__.py
|   |-- classification.py          # VGG11Classifier (Task 1)
|   |-- layers.py                  # CustomDropout layer
|   |-- localization.py            # VGG11Localizer (Task 2)
|   |-- multitask.py               # Shared-backbone multitask model (Task 4)
|   |-- segmentation.py            # VGG11UNet + Dice/BCE-Dice losses (Task 3)
|   |-- vgg11.py                   # VGG11 backbone (with optional BatchNorm)
```

## Data Source

This repository uses the **Oxford-IIIT Pet Dataset**.

- Dataset page: <https://www.robots.ox.ac.uk/~vgg/data/pets/>
- Local data root used in code: `data/`
- Relevant annotation sources:
    - `data/annotations/list.txt` for class mapping
    - `data/annotations/trainval.txt` and `test.txt` for splits
    - `data/annotations/xmls/` for bounding boxes
    - `data/annotations/trimaps/` for segmentation masks

The dataset loader (`data/pets_dataset.py`) performs:

- Deterministic train/val split (stratified by breed)
- Image resizing to `224x224`
- ImageNet normalization
- Classification-only train augmentations (flip + color jitter variants)
- Bounding box conversion to `[x_center, y_center, width, height]` in resized image pixel space
- Trimap remap from `{1,2,3}` to `{0,1,2}`

## Model Architecture

### 1. Backbone (`models/vgg11.py`)

- VGG11 configuration A
- Optional BatchNorm
- Adaptive average pooling to `7x7`
- Fully connected classifier stack with `CustomDropout`

### 2. Classification Model (`models/classification.py`)

- Class: `VGG11Classifier`
- Output: logits over 37 breeds
- Loss in training: `CrossEntropyLoss`

### 3. Localization Model (`models/localization.py`)

- Class: `VGG11Localizer`
- Uses VGG11 feature extractor + regression head
- Output format: `[x_center, y_center, width, height]`
- Supports transfer learning from classifier checkpoint
- Supports backbone freezing modes: `none`, `all`, `partial`

### 4. Segmentation Model (`models/segmentation.py`)

- Class: `VGG11UNet`
- Encoder: VGG11 feature pyramid
- Decoder: transposed convolutions + skip connections (`DoubleConv` blocks)
- Output: per-pixel logits with 3 classes
- Supports transfer learning and selective backbone freezing

### 5. Multitask Model (`models/multitask.py`)

- Class: `MultiTaskPerceptionModel`
- Shared VGG11 encoder + three task heads:
    - Classification head
    - Bounding box regression head
    - Segmentation decoder/head
- Loads task-specific checkpoints (`classifier.pth`, `localizer.pth`, `unet.pth`) to initialize shared model components
- Root-level import compatibility via `multitask.py`

## Loss Functions

### Localization Loss

Defined in `train.py` (Task 2):

- `MSELoss(pred_bbox, gt_bbox)`
- `IoULoss(pred_bbox, gt_bbox)` from `losses/iou_loss.py`
- Total loss: `L_loc = L_MSE + L_IoU`

`IoULoss` details:

- Input format: `[x_center, y_center, width, height]`
- IoU clamped to `[0, 1]`
- Loss = `1 - IoU`
- Supported reductions: `mean` (default), `sum`, `none`

### Segmentation Loss

Defined inside `VGG11UNet` as `BCEDiceLoss`:

- BCEWithLogits component for pixel-level supervision
- Dice component for overlap quality
- Weighted sum (default 0.5 BCE + 0.5 Dice)

### Classification Loss

- `CrossEntropyLoss` on breed logits.

## Training Pipeline

All training is handled by `train.py` with argument `--task`:

- `--task 1`: Classification training (`checkpoints/classifier.pth`)
- `--task 2`: Localization training (`checkpoints/localizer.pth`)
- `--task 3`: Segmentation training (`checkpoints/unet.pth`)

Common training features:

- Optimizer: `AdamW`
- Scheduler: `ReduceLROnPlateau`
- Optional mixed precision (`--amp`)
- W&B logging for train/validation metrics
- Configurable dropout, workers, batch size, and freeze strategy

### Quick Start Commands

```bash
# 1) Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2) Train task-wise
python train.py --task 1 --epochs 40 --batch_size 32 --pin_memory
python train.py --task 2 --epochs 40 --batch_size 8 --pin_memory --pretrained_vgg_path checkpoints/classifier.pth
python train.py --task 3 --epochs 40 --batch_size 4 --pin_memory --pretrained_vgg_path checkpoints/classifier.pth


```

## Multitask and Other Scripts

### Core Multitask Entry

- `multitask.py`: exposes `MultiTaskPerceptionModel` from `models.multitask` for autograder compatibility.

### Experiment/Analysis Scripts

- `scripts/visualize_features.py`
    - Logs early and deep convolution feature maps from the classifier.
- `scripts/plot_activations.py`
    - Compares conv3 activation distributions for BN vs no-BN setup.
- `scripts/eval_and_log.py`
    - Evaluates localization IoU and segmentation metrics and logs visual tables.
- `scripts/log_seg_metrics.py`
    - Runs segmentation validation and logs Dice, pixel accuracy, and val loss.
- `scripts/wild_inference.py`
    - Runs multitask predictions on `data/wild_images/` and logs bbox + segmentation overlays.
- `scripts/retrain1.sh`
    - Retrains classification variants for quick ablation reruns.

## Checkpoints

Expected filenames in `checkpoints/`:

- `classifier.pth`
- `localizer.pth`
- `unet.pth`

These are consumed by the multitask model initialization and grading pipeline.

## Report Links

- W&B Report: [DA6401 Assignment 2 Report](https://api.wandb.ai/links/be22b022-indian-institute-of-technology-madras/praemu7o)

- GitHub Repo Link : [GitHub Repo](https://github.com/n0xet3rnal/da6401_assignment_2.git)

## Submission Details

- Name: Jerry Jose
- Roll Number: BE22B022