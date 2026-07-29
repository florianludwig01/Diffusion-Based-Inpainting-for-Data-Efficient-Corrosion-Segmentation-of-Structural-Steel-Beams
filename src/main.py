import argparse

from training.train import DINO_train, YOLO_train
from testing.test import DINO_test, YOLO_test


EXPERIMENT_SEEDS = [42, 123, 456, 789, 1024, 2024, 31415, 27182, 65537, 99991,]


def parse_args():
    parser = argparse.ArgumentParser(description="Rust Segmentation Training")

    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--model", choices=["dino", "yolo"], required=True)

    # Paths
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--test_dir", type=str, default="/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/01_eigener_Datensatz/Validierung Rost")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--model_path", type=str)

    # Dataset name
    parser.add_argument("--dataset", type=str, default="R1")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # YOLO
    parser.add_argument("--yolo_model", type=str, default="yolo26s-seg.pt")
    # DINO
    parser.add_argument("--dino_model", type=str, default="dinov3")

    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075,
                 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                 0.6, 0.7, 0.8, 0.9]
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        for run_number, seed in enumerate(EXPERIMENT_SEEDS, start=1):
            args.seed = seed
            print(f"\nRun {run_number}/{len(EXPERIMENT_SEEDS)} | Seed: {seed}")

            if args.model == "dino":
                trainer = DINO_train(args)
                trainer.train(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    LR=args.lr,
                    weight_decay=args.weight_decay,
                    freeze_encoder=args.freeze_encoder,
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_min_delta=args.early_stopping_min_delta,
                    lr_scheduler_patience=args.lr_scheduler_patience,
                    lr_scheduler_factor=args.lr_scheduler_factor,
                    min_lr=args.min_lr
                )
            elif args.model == "yolo":
                trainer = YOLO_train(args)
                trainer.train(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    LR=args.lr,
                    weight_decay=args.weight_decay
                )
            else:
                raise ValueError(f"Unknown model: {args.model}")
    elif args.mode == "test":
        tester = DINO_test(args) if args.model == "dino" else YOLO_test(args)
        tester.test()


if __name__ == "__main__":
    main()
