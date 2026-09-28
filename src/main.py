import argparse

from testing.seed_evaluation import run_test
from training.train import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Corrosion Segmentation Training")

    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--model", choices=["dino", "yolo"], required=True)

    # Paths
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--test_dir", type=str, default="/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/01_eigener_Datensatz/Validierung Rost")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--model_path", type=str)
    parser.add_argument(
        "--models_dir",
        type=str,
        help="Directory with seed_*/best.pth or seed_*/best.pt",
    )
    parser.add_argument(
        "--all_seeds",
        action="store_true",
        help="Test and aggregate all models within --models_dir",
    )

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
        run_training(args)
    elif args.mode == "test":
        run_test(args)


if __name__ == "__main__":
    main()
