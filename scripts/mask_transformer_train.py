import numpy as np
import torch


from config import (
    CACHE_DIR,
    DEVICE,
    MASK_TRANSFORMER_PATH,
    MOTION_FEATS_DIR,  # ty:ignore[possibly-missing-import]
    NORM_PATH,
    OUTPUT_ROOT,
    RVQ_VAE_PATH,
    TRANSFORMER_CONFIG,
    VAE_CONFIG,
    DATASET_ROOT,
    IS_KAGGLE,
    NEW_MODELS_DIR,
)
from data.dataset import TokenizedMotionDataset, collate_fn
from models.residual_transformer import ResidualTransformer
from models.rvq_vae import load_rvq_vae, tokenize_all_motions
from models.transformer import MaskTransformer
from scripts.train_transformer import train_epoch, validate

if not IS_KAGGLE:
    print("[ERROR] This script is designed to run on Kaggle.")
    exit(1)

# ============================================================
# Training History Storage
# ============================================================
training_history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "epochs": [],
}

# ============================================================
# Initialize MaskTransformer
# ============================================================
print("Initializing MaskTransformer...")
transformer = MaskTransformer(
    num_tokens=int(VAE_CONFIG["num_embeddings"]),  # VAE codebook size
    code_dim=int(VAE_CONFIG["latent_dim"]),
    latent_dim=int(TRANSFORMER_CONFIG["latent_dim"]),
    ff_size=int(TRANSFORMER_CONFIG["ff_size"]),
    num_layers=int(TRANSFORMER_CONFIG["num_layers"]),
    num_heads=int(TRANSFORMER_CONFIG["num_heads"]),
    dropout=float(TRANSFORMER_CONFIG["dropout"]),
    clip_dim=512,
    clip_version="ViT-B/32",
    cond_drop_prob=0.1,
    device="cuda",
    max_seq_len=int(TRANSFORMER_CONFIG["max_token_len"]),
).to(DEVICE)

# ============================================================
# Initialize ResidualTransformer
# ============================================================
print("Initializing ResidualTransformer...")
residual_transformer_model = ResidualTransformer(
    num_tokens=int(VAE_CONFIG["num_embeddings"]),
    code_dim=int(VAE_CONFIG["latent_dim"]),
    latent_dim=int(TRANSFORMER_CONFIG["latent_dim"]),
    ff_size=int(TRANSFORMER_CONFIG["ff_size"]),
    num_layers=int(TRANSFORMER_CONFIG["num_layers"]),
    num_heads=int(TRANSFORMER_CONFIG["num_heads"]),
    dropout=float(TRANSFORMER_CONFIG["dropout"]),
    clip_dim=512,
    clip_version="ViT-B/32",
    cond_drop_prob=0.1,
    device="cuda",
    max_seq_len=int(TRANSFORMER_CONFIG["max_token_len"]),
    num_quantizers=int(VAE_CONFIG["num_quantizers"]),
).to(DEVICE)


motion_feats = list(MOTION_FEATS_DIR.glob("*.npy"))

if len(motion_feats) == 0:
    raise FileNotFoundError(f"No motion features found in {MOTION_FEATS_DIR}")

print(f"✓ Found {len(motion_feats)} motion feature files")
print(f"  Location: {MOTION_FEATS_DIR}")

# ============================================================
# Load Normalization Statistics (from pre-computed file)
# ============================================================
print("Loading normalization statistics...")
if NORM_PATH.exists():
    norm_data = np.load(NORM_PATH)
    mean = norm_data["mean"]
    std = norm_data["std"]
    print(f"   [OK] Loaded from {NORM_PATH}")
    print(f"   Shape: mean={mean.shape}, std={std.shape}")
else:
    # Fallback: compute from data (slower)
    print("   [WARN] Normalization file not found, computing from data...")
    all_files = list(MOTION_FEATS_DIR.glob("*.npy"))
    sample_data = [np.load(f) for f in all_files[:500]]
    all_feats = np.concatenate(sample_data, axis=0)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    del all_feats, sample_data

# ============================================================
# Load RVQ-VAE Model
# ============================================================
# Load the pretrained RVQ-VAE
vae_model, config = load_rvq_vae(RVQ_VAE_PATH, device="cuda")

# ============================================================
# Tokenize Dataset (or load from cache)
# ============================================================
token_cache_path = CACHE_DIR / "tokens.npy"
print("\nTokenizing dataset...")

token_data = tokenize_all_motions(
    vq_model=vae_model,
    motion_dir_=str(MOTION_FEATS_DIR),
    metadata_dir_=str(DATASET_ROOT),
    mean=mean,
    std=std,
    device=DEVICE,
    cache_path=str(token_cache_path),
)

print(f"   [OK] Tokenized {len(token_data)} sequences")

# ============================================================
# NOTE: ResidualTransformer will be loaded AFTER the MaskTransformer
# This cell is a placeholder - actual loading happens in Cell 17
# ============================================================
print(
    "[INFO] ResidualTransformer will be loaded after MaskTransformer (see next training section)"
)
residual_transformer_model = None  # Will be initialized later


USE_PRETRAINED = False
# ============================================================
# Load or Train
# ============================================================
if USE_PRETRAINED:
    # Load pre-trained weights
    print("\n[SKIP] Training - loading pre-trained models...")

    # Load MaskTransformer
    print(f"   Loading MaskTransformer from {MASK_TRANSFORMER_PATH}")
    checkpoint = torch.load(
        MASK_TRANSFORMER_PATH, map_location=DEVICE, weights_only=False
    )

    # Handle different checkpoint formats
    if "base_model_state_dict" in checkpoint:
        transformer.load_state_dict(checkpoint["base_model_state_dict"])
        print("   [OK] MaskTransformer loaded (from base_model_state_dict)")
    elif "model_state_dict" in checkpoint:
        transformer.load_state_dict(checkpoint["model_state_dict"])
        print("   [OK] MaskTransformer loaded (from model_state_dict)")
    else:
        transformer.load_state_dict(checkpoint)
        print("   [OK] MaskTransformer loaded (direct state dict)")

    # Load ResidualTransformer if available
    if "residual_model_state_dict" in checkpoint:
        residual_transformer_model.load_state_dict(
            checkpoint["residual_model_state_dict"]
        )
        print("   [OK] ResidualTransformer loaded (from checkpoint)")
    else:
        print(
            "   [WARN] ResidualTransformer weights not in checkpoint - using random init"
        )
        print("      (Generation will work but residual layers may be less refined)")

    transformer.eval()
    residual_transformer_model.eval()

else:
    # Train from scratch
    from functools import partial

    from torch.utils.data import DataLoader

    print("\n[TRAIN] Training Transformer from scratch...")

    # Prepare Datasets
    keys = list(token_data.keys())
    split_idx = int(len(keys) * 0.9)
    train_keys, val_keys = keys[:split_idx], keys[split_idx:]
    train_data = {k: token_data[k] for k in train_keys}
    val_data = {k: token_data[k] for k in val_keys}

    train_dataset = TokenizedMotionDataset(
        train_data,
        text_source=str(TRANSFORMER_CONFIG["text_source"]),
        max_token_len=int(TRANSFORMER_CONFIG["max_token_len"]),
    )
    val_dataset = TokenizedMotionDataset(
        val_data,
        text_source=str(TRANSFORMER_CONFIG["text_source"]),
        max_token_len=int(TRANSFORMER_CONFIG["max_token_len"]),
    )

    collate_fn_p = partial(collate_fn, return_all_layers=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(TRANSFORMER_CONFIG["batch_size"]),
        shuffle=True,
        collate_fn=collate_fn_p,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(TRANSFORMER_CONFIG["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn_p,
    )

    optimizer = torch.optim.AdamW(
        transformer.parameters(), lr=float(TRANSFORMER_CONFIG["lr"])
    )

    # Training Loop
    print(f"   Training for {TRANSFORMER_CONFIG['epochs']} epochs...")
    for epoch in range(1, int(TRANSFORMER_CONFIG["epochs"]) + 1):
        metrics = train_epoch(
            base_model=transformer,
            dataloader=train_loader,
            base_optimizer=optimizer,
            base_scheduler=None,
            device=DEVICE,
            epoch=epoch,
            writer=None,
        )

        val_metrics = validate(
            model=transformer, dataloader=val_loader, device=DEVICE, epoch=epoch
        )

        # Store metrics in training history
        # Note: train_epoch returns 'accuracy', validate returns 'acc'
        train_acc = metrics.get("accuracy", metrics.get("acc", 0.0))
        val_acc = val_metrics.get("acc", val_metrics.get("accuracy", 0.0))

        training_history["epochs"].append(epoch)
        training_history["train_loss"].append(metrics["loss"])
        training_history["train_acc"].append(train_acc)
        training_history["val_loss"].append(val_metrics["loss"])
        training_history["val_acc"].append(val_acc)

        print(
            f"   Epoch {epoch}: Train Loss {metrics['loss']:.4f}, Train Acc {train_acc:.2%}, Val Loss {val_metrics['loss']:.4f}, Val Acc {val_acc:.2%}"
        )

    # Save Model
    save_path = NEW_MODELS_DIR / "mask_transformer" / "best_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(transformer.state_dict(), save_path)
    print(f"   [OK] Model saved to {save_path}")

# ============================================================
# Display Training Summary
# ============================================================
if training_history["epochs"]:
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total epochs: {len(training_history['epochs'])}")
    print(f"  Final train loss: {training_history['train_loss'][-1]:.4f}")
    print(f"  Final train acc:  {training_history['train_acc'][-1]:.2%}")
    print(f"  Final val loss:   {training_history['val_loss'][-1]:.4f}")
    print(f"  Final val acc:    {training_history['val_acc'][-1]:.2%}")
    print(f"  Best val loss:    {min(training_history['val_loss']):.4f}")
    print(f"  Best val acc:     {max(training_history['val_acc']):.2%}")
    print("=" * 60)

    # Plot training curves
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(
        training_history["epochs"],
        training_history["train_loss"],
        label="Train Loss",
        marker="o",
    )
    axes[0].plot(
        training_history["epochs"],
        training_history["val_loss"],
        label="Val Loss",
        marker="s",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(
        training_history["epochs"],
        training_history["train_acc"],
        label="Train Acc",
        marker="o",
    )
    axes[1].plot(
        training_history["epochs"],
        training_history["val_acc"],
        label="Val Acc",
        marker="s",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "training_curves.png", dpi=150)
    plt.show()
    print(f"\n[OK] Training curves saved to {OUTPUT_ROOT / 'training_curves.png'}")

print("\n[OK] Transformers ready for inference!")
