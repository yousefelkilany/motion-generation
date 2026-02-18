# ============================================================
# SUBMISSION GENERATION (CSV-based for Kaggle)
# ============================================================
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import DEVICE, IS_KAGGLE, RVQ_VAE_PATH, TRANSFORMER_CONFIG, VAE_CONFIG
from models.residual_transformer import ResidualTransformer
from models.rvq_vae import load_rvq_vae
from models.transformer import MaskTransformer

gc.collect()
torch.cuda.empty_cache()

# ============================================================
# Configuration
# ============================================================

SUBMISSION_CONFIG = {
    "text_source": "both",  # 'sentence', 'gloss', or 'both'
    "timesteps": 10,  # Number of demasking timesteps
    "cond_scale": 4.0,  # Classifier-free guidance scale
    "temperature": 1.0,  # Sampling temperature
    "default_token_length": 50,  # Default token length if estimation fails
    "use_residual": True,  # Set to False to disable residual layers entirely
}


# ============================================================
# Helper Functions
# ============================================================


def estimate_token_length(text: str, default_length: int = 50) -> int:
    """
    Simple heuristic to estimate motion token length from text.
    Can be replaced with a trained length estimator.
    """
    # Rough estimate: ~8 tokens per word, capped between 20-200
    word_count = len(text.split())
    estimated = min(max(word_count * 8, 20), 200)
    return estimated


def tokens_to_string(tokens) -> str:
    """Convert tensor/array of token IDs to space-separated string."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    return " ".join(map(str, tokens.astype(int).flatten().tolist()))


def generate_tokens_for_text(
    text: str,
    transformer_model,
    vae_model,
    residual_model=None,
    device="cuda",
    config=None,
):
    """
    Generate all layer tokens for a single text prompt.

    Returns:
        dict with keys: base_tokens, residual_1, ..., residual_{num_quantizers-1}
        Number of residual layers is dynamic based on vae_model.num_quantizers
    """
    if config is None:
        config = SUBMISSION_CONFIG

    if config is None:
        raise ValueError("No configuration provided")

    # Estimate token length
    token_len = estimate_token_length(text, int(config["default_token_length"]))
    m_lens = torch.tensor([token_len], device=device)

    num_quantizers = vae_model.num_quantizers
    num_residual_layers = num_quantizers - 1  # Dynamic based on VAE
    max_token_id = vae_model.rvq.quantizers[0].num_embeddings - 1

    with torch.no_grad():
        # 1. Generate base layer tokens
        base_tokens = transformer_model.generate(
            texts=[text],
            m_lens=m_lens,
            timesteps=config["timesteps"],
            cond_scale=config["cond_scale"],
            temperature=config["temperature"],
        )

        # Clamp to valid range
        base_tokens = torch.clamp(base_tokens, min=0, max=max_token_id)
        base_tokens_np = base_tokens[0].cpu().numpy()  # (seq_len,)

        # 2. Generate residual layers (dynamic count)
        residual_tokens_list = []

        if config.get("use_residual", True) and residual_model is not None:
            # Use residual transformer
            all_layer_tokens = [base_tokens]

            for layer_idx in range(1, num_quantizers):
                try:
                    res_tokens = residual_model.generate_layer(
                        prev_layer_tokens=all_layer_tokens,
                        layer_idx=layer_idx,
                        texts=[text],
                        m_lens=m_lens,
                        vq_model=vae_model,
                        cond_scale=config["cond_scale"],
                        temperature=config["temperature"],
                    )
                    res_tokens = torch.clamp(res_tokens, min=0, max=max_token_id)
                    all_layer_tokens.append(res_tokens)
                    residual_tokens_list.append(res_tokens[0].cpu().numpy())
                except Exception:
                    # Fallback to zeros if generation fails
                    residual_tokens_list.append(np.zeros_like(base_tokens_np))
        else:
            # No residual model or disabled - use zeros
            for _ in range(num_residual_layers):
                residual_tokens_list.append(np.zeros_like(base_tokens_np))

    # Build result dictionary with dynamic residual layers
    result = {
        "base_tokens": base_tokens_np,
        "num_residual_layers": num_residual_layers,  # Store for reference
    }

    # Add residual layers dynamically based on num_quantizers
    for i in range(num_residual_layers):
        key = f"residual_{i + 1}"
        if i < len(residual_tokens_list):
            result[key] = residual_tokens_list[i]
        else:
            result[key] = np.zeros_like(base_tokens_np)

    return result


def is_running_in_notebook():
    try:
        from IPython import get_ipython  # type: ignore[unresolved-import]

        if "IPKernelApp" in get_ipython().config:
            return True
        # Check for terminal IPython as well
        if "terminal" in str(type(get_ipython())):
            return False
    except NameError:
        # get_ipython is not defined, so not in a notebook/IPython environment
        return False
    return False


# ============================================================
# Main Submission Generation (CSV-based)
# ============================================================
def generate_submission_from_csv(
    test_csv_path: Path,
    transformer_model,
    vae_model,
    residual_model=None,
    output_path: str = "submission.csv",
    config=None,
):
    """
    Generate submission CSV from test.csv file.

    test.csv format:
        id, sentence, gloss
        6420249, "Mary never told me...", "ME NEVER TOLD..."
    """
    if config is None:
        config = SUBMISSION_CONFIG

    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    # Load test CSV
    test_df = pd.read_csv(test_csv_path)
    print(f"\nLoaded test CSV: {len(test_df)} samples")
    print(f"Columns: {list(test_df.columns)}")

    results = []

    for idx, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Generating submission"
    ):
        seq_id = row["id"]
        sentence = str(row.get("sentence", "")) if pd.notna(row.get("sentence")) else ""
        gloss = str(row.get("gloss", "")) if pd.notna(row.get("gloss")) else ""

        # Select text source
        text_source = config["text_source"]

        if text_source == "both":
            # Combine sentence and gloss
            if sentence and gloss:
                text = f"{sentence} {gloss}"
            else:
                text = sentence or gloss
        elif text_source == "sentence":
            text = sentence or gloss  # Fallback to gloss
        elif text_source == "gloss":
            text = gloss or sentence  # Fallback to sentence
        else:
            text = sentence or gloss

        if not text.strip():
            print(f"⚠ Warning: No text found for {seq_id}, using placeholder")
            text = "sign language motion"

        try:
            # Generate tokens
            tokens_dict = generate_tokens_for_text(
                text=text,
                transformer_model=transformer_model,
                vae_model=vae_model,
                residual_model=residual_model,
                device=DEVICE,
                config=config,
            )

            # Build CSV row
            result_row = {
                "id": seq_id,  # Match test.csv column name
                "base_tokens": tokens_to_string(tokens_dict["base_tokens"]),
            }

            # Add residual columns dynamically
            num_residual = tokens_dict.get(
                "num_residual_layers", vae_model.num_quantizers - 1
            )
            for i in range(num_residual):
                key = f"residual_{i + 1}"
                result_row[key] = tokens_to_string(tokens_dict.get(key, np.zeros(1)))

            results.append(result_row)

        except Exception as e:
            print(f"⚠ Error generating tokens for {seq_id}: {e}")
            # Add placeholder row to maintain alignment
            result_row = {"id": seq_id, "base_tokens": "0 " * 50}
            num_residual = vae_model.num_quantizers - 1
            for i in range(num_residual):
                result_row[f"residual_{i + 1}"] = "0 " * 50
            results.append(result_row)
            continue

    if len(results) == 0:
        raise ValueError("No sequences processed successfully!")

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"\n{'=' * 50}")
    print("SUBMISSION GENERATED")
    print(f"{'=' * 50}")
    print(f"  Output file: {output_path}")
    print(f"  Total sequences: {len(results)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"{'=' * 50}")

    return df


if __name__ == "__main__":
    # Test CSV path (Kaggle)
    if IS_KAGGLE:
        TEST_CSV_PATH = Path(
            "/kaggle/input/motion-s-hierarchical-text-to-motion-generation-for-sign-language/test.csv"
        )
    else:
        TEST_CSV_PATH = Path("data/test.csv")  # Local testing

    print(f"Test CSV path: {TEST_CSV_PATH}")
    print(f"Test CSV exists: {TEST_CSV_PATH.exists()}")

    # ============================================================
    # Load RVQ-VAE Model
    # ============================================================
    # Load the pretrained RVQ-VAE
    vae_model, config = load_rvq_vae(RVQ_VAE_PATH, device="cuda")

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

    checkpoint_path = ""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if "base_model_state_dict" in checkpoint:
        transformer.load_state_dict(checkpoint["base_model_state_dict"])
        print("   [OK] MaskTransformer loaded (from base_model_state_dict)")
    elif "model_state_dict" in checkpoint:
        transformer.load_state_dict(checkpoint["model_state_dict"])
        print("   [OK] MaskTransformer loaded (from model_state_dict)")
    else:
        transformer.load_state_dict(checkpoint)
        print("   [OK] MaskTransformer loaded (direct state dict)")

    if "residual_model_state_dict" in checkpoint:
        residual_transformer_model.load_state_dict(
            checkpoint["residual_model_state_dict"]
        )

    # ============================================================
    # GENERATE SUBMISSION
    # ============================================================

    # Check if residual transformer is available
    has_residual = (
        "residual_transformer_model" in dir() and residual_transformer_model is not None
    )

    print("\n" + "=" * 60)
    print("SUBMISSION GENERATION")
    print("=" * 60)
    print(f"  Transformer model: {'✓ Loaded' if transformer else '✗ Missing'}")
    print(f"  VAE model: {'✓ Loaded' if vae_model else '✗ Missing'}")
    print(
        f"  Residual model: {'✓ Loaded' if has_residual else '✗ Not available (using zeros)'}"
    )
    print(f"  Text source: {SUBMISSION_CONFIG['text_source']}")
    print(f"  Use residual: {SUBMISSION_CONFIG['use_residual'] and has_residual}")
    print("=" * 60)

    # Generate submission from test CSV
    if TEST_CSV_PATH.exists():
        submission_df = generate_submission_from_csv(
            test_csv_path=TEST_CSV_PATH,
            transformer_model=transformer,
            vae_model=vae_model,
            residual_model=residual_transformer_model if has_residual else None,
            output_path="submission.csv",
            config=SUBMISSION_CONFIG,
        )

        # Display sample
        if is_running_in_notebook() and submission_df is not None:
            print("\nSample rows:")
            display(submission_df.head(3))  # noqa: F821  # type: ignore[unresolved-reference]
    else:
        print(f"\n⚠ Test CSV not found at: {TEST_CSV_PATH}")
        print("  Expected format:")
        print("    id,sentence,gloss")
        print("    6420249,Mary never told me...,ME NEVER TOLD...")

    # ============================================================
    # VERIFY SUBMISSION FILE
    # ============================================================
    from pathlib import Path

    submission_file = Path("submission.csv")

    if submission_file.exists():
        df = pd.read_csv(submission_file)

        print("✓ submission.csv successfully created!")
        print(f"\n  File size: {submission_file.stat().st_size / 1024:.2f} KB")
        print(f"  Sequences: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        # Count residual columns dynamically
        residual_cols = [c for c in df.columns if c.startswith("residual_")]
        print(f"  Residual layers: {len(residual_cols)}")

        # Sample token lengths
        if len(df) > 0:
            sample_base = df["base_tokens"].iloc[0]
            sample_len = len(sample_base.split())
            print(f"\n  Sample base token length: {sample_len} tokens")

            # Check if first residual layer contains zeros
            if residual_cols:
                sample_res = df[residual_cols[0]].iloc[0]
                res_tokens = list(map(int, sample_res.split()))
                all_zeros = all(t == 0 for t in res_tokens)
                print(
                    f"  Residuals are zeros: {'Yes' if all_zeros else 'No (using residual transformer)'}"
                )

        print("\n" + "=" * 50)
        print("Ready to submit! 🚀")
        print("=" * 50)
    else:
        print("⚠ submission.csv not found")
