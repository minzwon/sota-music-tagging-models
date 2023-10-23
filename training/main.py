import argparse
from pathlib import Path

from datasets import DATASETS
from solver import Solver


def main(config):
    solver = Solver(config)
    solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mtat",
        choices=list(DATASETS),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="fcn",
        choices=[
            "fcn",
            "musicnn",
            "crnn",
            "sample",
            "se",
            "short",
            "short_res",
            "attention",
            "hcnn",
        ],
    )
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_tensorboard", type=int, default=1)
    parser.add_argument("--model_save_path", type=str, default="./models")
    parser.add_argument("--model_load_path", type=str, default="./models")
    parser.add_argument("--load_model", type=int, default=1)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--log_step", type=int, default=20)

    config = parser.parse_args()

    if config.load_model:
        model_load_dir: Path = (
            Path(config.model_load_path) / config.dataset / config.model_type
        )
        model_load_dir.mkdir(parents=True, exist_ok=True)
        config.model_load_path = model_load_dir / "best_model.pth"

    model_save_dir: Path = (
        Path(config.model_save_path) / config.dataset / config.model_type
    )
    model_save_dir.mkdir(parents=True, exist_ok=True)
    config.model_save_path = model_save_dir / "best_model.pth"

    print(config)
    main(config)
