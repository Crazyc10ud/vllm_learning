from __future__ import annotations

import argparse
from pathlib import Path

from finetune_clip.training.pipeline import ExperimentConfig, ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP/SigLIP vision backbones.")
    parser.add_argument("--model-family", choices=["clip", "siglip"], required=True)
    parser.add_argument("--pretrained-name", type=str, required=True)
    parser.add_argument("--dataset", choices=["cifar100", "isic2018"], required=True)
    parser.add_argument(
        "--mode",
        choices=["full", "lora", "linear-probing"],
        default="full",
        help="Fine-tuning strategy.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        model_family=args.model_family,
        pretrained_name=args.pretrained_name,
        dataset=args.dataset,
        finetune_mode=args.mode,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(config)
    metrics = runner.run()
    print(f"Finished experiment. Test accuracy: {metrics['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
