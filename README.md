# CLIP & SigLIP Fine-tuning Toolkit

This project demonstrates how to fine-tune CLIP (ViT patch16) and SigLIP2 vision backbones on both the CIFAR-100 and ISIC-2018 datasets using three different strategies:

1. **Full fine-tuning** of all backbone parameters.
2. **LoRA fine-tuning** with a hand-written Low-Rank Adaptation implementation.
3. **Linear probing** where only the classification head is trained.

The training script logs loss/accuracy curves, saves the resulting checkpoints, and produces class activation maps (CAM) before and after training so you can visually inspect what each model attends to.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running experiments

The main entry point is `scripts/train_experiment.py`. Supply the model family, pretrained name, dataset, fine-tuning strategy, and where to write artifacts.

```bash
python scripts/train_experiment.py \
  --model-family clip \
  --pretrained-name openai/clip-vit-base-patch16 \
  --dataset cifar100 \
  --mode lora \
  --output-dir outputs/clip_cifar100_lora \
  --epochs 10 \
  --batch-size 64
```

To run the SigLIP2 counterpart simply swap the arguments:

```bash
python scripts/train_experiment.py \
  --model-family siglip \
  --pretrained-name google/siglip2-base-patch16-256 \
  --dataset isic2018 \
  --mode full \
  --output-dir outputs/siglip_isic_full
```

Each invocation produces:

* `training_metrics.png` – side-by-side plots of training/validation loss and validation accuracy.
* `metrics.json` – epoch-wise metrics and the final test accuracy.
* `model.pt` – the fine-tuned model weights.
* `cam/before_cam.png` and `cam/after_cam.png` – CAM overlays computed with a custom Grad-CAM implementation.

## Datasets

* **CIFAR-100** is downloaded automatically through `torchvision`.
* **ISIC-2018** is automatically fetched from the official challenge URLs. The script stores the dataset under `data/isic2018` and creates deterministic train/validation/test splits.

## Hand-written components

* `finetune_clip/training/lora.py` implements LoRA injection without relying on external libraries.
* `finetune_clip/visualization/cam.py` contains a ViT-specific Grad-CAM routine for visualizing attention before and after training.

Feel free to adjust hyperparameters or extend the scripts for additional datasets and backbones.
