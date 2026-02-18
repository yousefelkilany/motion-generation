from pathlib import Path
import torch

# Base Paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "data" / "tokenized_cache"

# Kaggle Detection
IS_KAGGLE = Path("/kaggle/working").exists()

KAGGLE_INPUT = Path("/kaggle/input")

# ============================================================
# MODE SELECTION
# ============================================================
USE_PRETRAINED = True  # True = Quick Start, False = Full Training

# ============================================================
# PATHS
# ============================================================
if IS_KAGGLE:
    DATA_BASE = (
        KAGGLE_INPUT
        / "competitions"
        / "motion-s-hierarchical-text-to-motion-generation-for-sign-language"
    )
    DATASET_ROOT = DATA_BASE / "Train"
    MOTION_FEATS_DIR = DATA_BASE / "Motion-Features"
    NORM_PATH = DATA_BASE / "normalization.npz"  # Add this file to your dataset

    # Pre-trained models
    PRETRAINED_RVQ_VAE_PATH = (
        KAGGLE_INPUT
        / "models/antonygithinji"
        / "motion-s-vae-rvq"
        / "pytorch"
        / "default"
        / "3"
        / "rvq_vae_best.pth"
    )
    PRETRAINED_TRANSFORMER_PATH = (
        KAGGLE_INPUT
        / "motion-s-base-gen"
        / "pytorch"
        / "default"
        / "1"
        / "best_model.pth"
    )
    PRETRAINED_LENGTH_ESTIMATOR_PATH = (
        KAGGLE_INPUT
        / "models/antonygithinji"
        / "motion-s-length-estimator"
        / "pytorch"
        / "default"
        / "1"
        / "length_estimator_best.pth"
    )
    PRETRAINED_ALIGNMENT_PATH = (
        KAGGLE_INPUT
        / "models/antonygithinji"
        / "motion-s-evaluator-t2m"
        / "pytorch"
        / "default"
        / "4"
        / "Public_t2m_align.pth"
    )

    # Output
    OUTPUT_ROOT = PROJECT_ROOT / "output"
    NEW_MODELS_DIR = OUTPUT_ROOT / "models"
    CACHE_DIR = PROJECT_ROOT / "tokenized_cache"
else:
    raise Exception("Currently, code is not ready for local use")
    DATASET_ROOT = PROJECT_ROOT / "dataset"
    NORM_PATH = PROJECT_ROOT / "data" / "motion-s_feats" / "normalization.npz"
    PRETRAINED_RVQ_VAE_PATH = PROJECT_ROOT / "models" / "rvq_vae_best.pth"
    PRETRAINED_TRANSFORMER_PATH = PROJECT_ROOT / "models" / "best_model.pth"
    OUTPUT_ROOT = PROJECT_ROOT / "notebook_output"
    NEW_MODELS_DIR = OUTPUT_ROOT / "models"
    CACHE_DIR = PROJECT_ROOT / "data" / "tokenized_cache"

# Model selection
if USE_PRETRAINED:
    RVQ_VAE_PATH = PRETRAINED_RVQ_VAE_PATH
    MASK_TRANSFORMER_PATH = PRETRAINED_TRANSFORMER_PATH
else:
    RVQ_VAE_PATH = NEW_MODELS_DIR / "vae" / "best_model.pth"
    MASK_TRANSFORMER_PATH = NEW_MODELS_DIR / "mask_transformer" / "best_model.pth"


# Model Configs
VAE_CONFIG = {
    "latent_dim": 256,
    "num_layers": 4,
    "num_quantizers": 6,
    "num_embeddings": 512,
    "downsampling_ratio": 4,
}

TRANSFORMER_CONFIG = {
    "latent_dim": 384,
    "ff_size": 1024,
    "num_layers": 8,  # Match trained model
    "num_heads": 6,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 2e-4,
    "epochs": 2,  # Increase for better results
    "max_token_len": 800,  # Match trained model
    "text_source": "both",  # 'sentence' or 'gloss'
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # Create directories
    for p in [OUTPUT_ROOT, NEW_MODELS_DIR, CACHE_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    # Verify
    print(f"Dataset: {DATASET_ROOT} - {'✓' if DATASET_ROOT.exists() else '✗'}")
    print(
        f"Normalization: {NORM_PATH} - {'✓' if NORM_PATH.exists() else '✗ Upload needed'}"
    )
    print(
        f"Motion Features: {MOTION_FEATS_DIR} - {'✓' if MOTION_FEATS_DIR.exists() else '✗'}"
    )
    print(f"RVQ-VAE: {RVQ_VAE_PATH} - {'✓' if RVQ_VAE_PATH.exists() else '✗'}")
    print(
        f"Transformer: {MASK_TRANSFORMER_PATH} - {'✓' if MASK_TRANSFORMER_PATH.exists() else '✗'}"
    )
