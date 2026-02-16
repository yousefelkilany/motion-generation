import argparse
from data.motion_process import batch_process_bvh
from config import DATASET_ROOT, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="Process BVH files into motion features."
    )
    parser.add_argument(
        "--src",
        type=str,
        default=str(DATASET_ROOT),
        help="Source directory for BVH files",
    )
    parser.add_argument(
        "--tgt",
        type=str,
        default=str(PROJECT_ROOT / "data" / "motion_feats"),
        help="Target directory for processed features",
    )
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    args = parser.parse_args()

    print(f"Processing BVH files from {args.src} to {args.tgt} at {args.fps} FPS...")
    batch_process_bvh(args.src, args.tgt, target_fps=args.fps, overwrite=True)
    print("Processing complete.")


if __name__ == "__main__":
    main()
