import argparse
import torch
from config import VAE_CONFIG, TRANSFORMER_CONFIG, DEVICE


def main():
    parser = argparse.ArgumentParser(description="Motion-S: Text-to-Sign Generation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "process", "infer"],
        default="infer",
        help="Running mode",
    )
    args = parser.parse_args()

    if args.mode == "process":
        from scripts.process_data import main as process_main

        process_main()
    elif args.mode == "train":
        print("Training mode selected. (Implementation in progress)")
        # Call training logic here
    elif args.mode == "infer":
        print("Inference mode selected. (Implementation in progress)")
        # Call inference logic here


if __name__ == "__main__":
    main()
