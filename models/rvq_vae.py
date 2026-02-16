import os
from data.dataset import parse_metadata
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    """
    1D Convolutional Encoder for motion sequences.
    Input: (B, D_POSE, N) - treats D_POSE as channels, N as sequence length.
    Output: (B, d, n) - downsampled latent with n = N // downsampling_ratio.
    Uses stacked Conv1D with stride=downsampling_factor for downsampling.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        downsampling_ratio: int = 4,
        num_layers: int = 4,
        hidden_dim: int = 512,
    ):
        super(MotionEncoder, self).__init__()
        self.input_dim: int = input_dim
        self.latent_dim: int = latent_dim
        self.downsampling_ratio: int = downsampling_ratio
        self.stride: int = 2  # Assuming binary downsampling (stride=2 per layer)
        self.num_layers: int = num_layers

        # Build conv layers
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_dim
            stride = self.stride if i < int(np.log2(downsampling_ratio)) else 1
            layers.extend(
                [
                    nn.Conv1d(
                        current_dim, out_dim, kernel_size=3, stride=stride, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(out_dim),  # Stabilizes training
                ]
            )
            current_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        self.downsampled_len: int = None  # Computed on first forward

    def forward(self, x):
        # x: (B, D_POSE, N)
        if self.downsampled_len is None:
            with torch.no_grad():
                dummy = self.encoder(x)
                self.downsampled_len = dummy.shape[-1]

        encoded = self.encoder(x)  # (B, latent_dim, n)
        return encoded  # Keep as (B, d, n); can transpose/reshape downstream if needed


class MotionDecoder(nn.Module):
    """
    1D Convolutional Decoder for motion sequences.
    Input: (B, d, n) - downsampled latent with n = N // downsampling_ratio.
    Output: (B, D_POSE, N) - upsampled to original motion dimension.
    Uses stacked Conv1D with transpose convolutions for upsampling.
    """

    def __init__(
        self,
        output_dim: int,
        latent_dim: int = 256,
        downsampling_ratio: int = 4,
        num_layers: int = 4,
        hidden_dim: int = 512,
    ):
        super(MotionDecoder, self).__init__()
        self.output_dim: int = output_dim
        self.latent_dim: int = latent_dim
        self.downsampling_ratio: int = downsampling_ratio
        self.stride: int = 2  # Binary upsampling (stride=2 per layer)
        self.num_layers: int = num_layers

        # Build conv transpose layers (mirror of encoder)
        layers = []
        current_dim = latent_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            stride = self.stride if i < int(np.log2(downsampling_ratio)) else 1
            # Use ConvTranspose1d for upsampling
            # output_padding compensates for stride to ensure exact upsampling
            output_padding = (stride - 1) if stride > 1 else 0
            layers.extend(
                [
                    nn.ConvTranspose1d(
                        current_dim,
                        out_dim,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        output_padding=output_padding,
                    ),
                    nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity(),
                    nn.BatchNorm1d(out_dim) if i < num_layers - 1 else nn.Identity(),
                ]
            )
            current_dim = out_dim

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, latent_dim, n)
        decoded = self.decoder(x)  # (B, D_POSE, N)
        return decoded


class VectorQuantizer(nn.Module):
    """
    Single-layer vector quantizer with EMA updates.
    Implements straight-through estimator for gradients.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 1.0,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim: int = embedding_dim
        self.num_embeddings: int = num_embeddings
        self.commitment_cost: float = commitment_cost
        self.decay: float = decay
        self.epsilon: float = epsilon

        # Initialize codebook
        self.embedding: torch.Tensor
        self.cluster_size: torch.Tensor
        self.embedding_avg: torch.Tensor
        self.register_buffer("embedding", torch.randn(embedding_dim, num_embeddings))
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer(
            "embedding_avg", torch.zeros(embedding_dim, num_embeddings)
        )

    def forward(self, inputs):
        # inputs: (B, d, n) - reshape to (B*n, d) for quantization
        B, d, n = inputs.shape
        flat_input = inputs.permute(0, 2, 1).contiguous()  # (B, n, d)
        flat_input = flat_input.reshape(-1, d)  # (B*n, d)

        # Calculate distances to codebook entries
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding**2, dim=0, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embedding)
        )  # (B*n, num_embeddings)

        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)  # (B*n,)

        # Quantize: replace with nearest codebook entry
        quantized = self.embedding[:, encoding_indices].t()  # (B*n, d)
        quantized = quantized.reshape(B, n, d).permute(0, 2, 1)  # (B, d, n)

        # Straight-through estimator: use quantized in forward, gradients pass through inputs
        quantized_st = inputs + (quantized - inputs).detach()

        # Commitment loss: encourage inputs to be close to quantized codes
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # EMA update (only during training)
        # EMA update (only during training)
        if self.training:
            with torch.no_grad():  # Critical: wrap entire EMA update
                # Update cluster sizes and embedding averages
                encodings = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).float()  # (B*n, num_embeddings)

                # Detach flat_input before using in EMA updates
                flat_input_detached = flat_input.detach()

                # Update cluster size
                self.cluster_size.mul_(self.decay).add_(
                    encodings.sum(0), alpha=1 - self.decay
                )

                # Update embedding averages (using detached input)
                embed_sum = flat_input_detached.t() @ encodings  # (d, num_embeddings)
                self.embedding_avg.mul_(self.decay).add_(
                    embed_sum, alpha=1 - self.decay
                )

                # Update codebook entries
                n_clusters = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.epsilon)
                    / (n_clusters + self.num_embeddings * self.epsilon)
                    * n_clusters
                )
                embed_normalized = self.embedding_avg / cluster_size.unsqueeze(0)
                self.embedding.copy_(embed_normalized)

        # Reshape encoding_indices back to (B, n)
        encoding_indices = encoding_indices.reshape(B, n)

        return quantized_st, encoding_indices, loss


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantization (RVQ) with V+1 quantization layers.
    Implements hierarchical quantization as described in MoMask paper.
    """

    def __init__(
        self,
        num_quantizers: int = 6,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 1.0,
        decay: float = 0.99,
        quantization_dropout: float = 0.2,
    ):
        super(ResidualVectorQuantizer, self).__init__()
        self.num_quantizers: int = num_quantizers  # V+1 total layers (including base)
        self.quantization_dropout: float = quantization_dropout

        # Create V+1 quantizers
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    commitment_cost=commitment_cost,
                    decay=decay,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, inputs, return_tokens=True):
        """
        Forward pass through residual quantization.

        Args:
            inputs: (B, d, n) - continuous latent sequence from encoder
            return_tokens: if True, return token indices; if False, only return quantized codes

        Returns:
            quantized: (B, d, n) - sum of all quantized codes
            tokens: List of (B, n) token sequences, one per layer
            commitment_loss: scalar - sum of commitment losses from all layers
        """
        B, d, n = inputs.shape

        # Initialize residual with input
        residual = inputs  # r^0 = b̃
        quantized_codes = []
        tokens = []
        commitment_losses = []

        # Determine how many layers to use (quantization dropout during training)
        num_active_layers: int = self.num_quantizers
        if self.training and self.quantization_dropout > 0:
            # Randomly disable last 0 to V layers
            if torch.rand(1).item() < self.quantization_dropout:
                num_active_layers = int(
                    torch.randint(0, self.num_quantizers, (1,)).item() + 1
                )

        # Quantize through each layer
        for v in range(num_active_layers):
            # Quantize current residual: b^v = Q(r^v)
            quantized, token_indices, loss = self.quantizers[v](residual)

            quantized_codes.append(quantized)
            tokens.append(token_indices)
            commitment_losses.append(loss)

            # Compute next residual: r^{v+1} = r^v - b^v
            if v < num_active_layers - 1:
                residual = residual - quantized

        # Pad with zeros if fewer layers were used
        while len(quantized_codes) < self.num_quantizers:
            quantized_codes.append(torch.zeros_like(inputs))
            tokens.append(torch.zeros((B, n), dtype=torch.long, device=inputs.device))

        # Sum all quantized codes: Σ_{v=0}^V b^v
        quantized = sum(quantized_codes)

        # Total commitment loss
        commitment_loss = sum(commitment_losses)

        if return_tokens:
            return quantized, tokens, commitment_loss
        else:
            return quantized, commitment_loss

    def quantize_from_tokens(self, tokens):
        """
        Reconstruct quantized codes from token indices.

        Args:
            tokens: List of (B, n) token sequences, one per layer

        Returns:
            quantized: (B, d, n) - sum of all quantized codes
        """
        quantized_codes = []

        for v, token_seq in enumerate(tokens):
            B, n = token_seq.shape
            d = self.quantizers[v].embedding.shape[0]

            # Lookup codebook entries
            # token_seq: (B, n) with indices in [0, num_embeddings-1]
            # embedding: (d, num_embeddings)
            # We want: (B, d, n)
            flat_tokens = token_seq.reshape(-1)  # (B*n,)
            quantized_flat = self.quantizers[v].embedding[:, flat_tokens]  # (d, B*n)
            quantized = quantized_flat.reshape(d, B, n).permute(1, 0, 2)  # (B, d, n)

            quantized_codes.append(quantized)

        # Sum all quantized codes
        quantized = sum(quantized_codes)
        return quantized


class RVQVAE(nn.Module):
    """
    Complete Residual VQ-VAE model combining encoder, RVQ, and decoder.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 256,
        downsampling_ratio: int = 4,
        num_layers: int = 4,
        hidden_dim: int = 512,
        num_quantizers: int = 6,
        num_embeddings: int = 512,
        commitment_cost: float = 1.0,
        decay: float = 0.99,
        quantization_dropout: float = 0.2,
    ):
        super(RVQVAE, self).__init__()

        self.encoder = MotionEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            downsampling_ratio=downsampling_ratio,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )

        self.rvq = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            quantization_dropout=quantization_dropout,
        )

        self.decoder = MotionDecoder(
            output_dim=output_dim,
            latent_dim=latent_dim,
            downsampling_ratio=downsampling_ratio,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.latent_dim: int = latent_dim
        self.downsampling_ratio: int = downsampling_ratio
        self.num_quantizers: int = num_quantizers

    def forward(self, x, return_tokens=True):
        """
        Forward pass through RVQ-VAE.

        Args:
            x: (B, D_POSE, N) - input motion sequence
            return_tokens: if True, return token sequences

        Returns:
            reconstructed: (B, D_POSE, N) - reconstructed motion
            tokens: List of (B, n) token sequences (if return_tokens=True)
            commitment_loss: scalar - commitment loss from RVQ
        """
        # Encode
        latent = self.encoder(x)  # (B, d, n)

        # Quantize
        quantized, tokens, commitment_loss = self.rvq(
            latent, return_tokens=return_tokens
        )

        # Decode
        reconstructed = self.decoder(quantized)  # (B, D_POSE, N)

        if return_tokens:
            return reconstructed, tokens, commitment_loss
        else:
            return reconstructed, commitment_loss

    def encode_to_tokens(self, x):
        """
        Encode motion to discrete tokens without decoding.

        Args:
            x: (B, D_POSE, N) - input motion sequence

        Returns:
            tokens: List of (B, n) token sequences
        """
        latent = self.encoder(x)  # (B, d, n)
        _, tokens, _ = self.rvq(latent, return_tokens=True)
        return tokens

    def decode_from_tokens(self, tokens):
        """
        Decode discrete tokens back to motion.

        Args:
            tokens: List of (B, n) token sequences

        Returns:
            reconstructed: (B, D_POSE, N) - reconstructed motion
        """
        quantized = self.rvq.quantize_from_tokens(tokens)  # (B, d, n)
        reconstructed = self.decoder(quantized)  # (B, D_POSE, N)
        return reconstructed


def compute_output_length(model, seq_len):
    """
    Compute the actual downsampled length by running a dummy forward on CPU.
    """
    model.eval()
    model.to("cpu")
    dummy_input = torch.randn(1, model.input_dim, seq_len)
    with torch.no_grad():
        output = model(dummy_input)
        return output.shape[-1]


def encode_motion(
    features, model, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Encode the (N, D_POSE) features to latent (n, d).
    Returns: latent_np (n, d) as numpy array.
    """
    model.eval()
    model.to(device)

    # Prepare input: (1, D_POSE, N)
    N, D = features.shape
    x = torch.from_numpy(features.T).unsqueeze(0).float().to(device)  # (1, D, N)

    with torch.no_grad():
        if isinstance(model, RVQVAE):
            # For RVQVAE, encode to tokens first, then get quantized latents
            tokens = model.encode_to_tokens(x)
            quantized = model.rvq.quantize_from_tokens(tokens)
            latent_np = quantized.squeeze(0).cpu().numpy().T  # (n, d)
        else:
            # For regular encoder
            latent = model(x)  # (1, d, n)
            latent_np = latent.squeeze(0).cpu().numpy().T  # (n, d)

    return latent_np


def decode_motion(latent, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Decode the (n, d) latent to motion features (N, D_POSE).
    Returns: motion_np (N, D_POSE) as numpy array.
    """
    model.eval()
    model.to(device)

    # Prepare input: (1, d, n)
    n, d = latent.shape
    x = torch.from_numpy(latent.T).unsqueeze(0).float().to(device)  # (1, d, n)

    with torch.no_grad():
        if isinstance(model, RVQVAE):
            # For RVQVAE, use decoder directly
            decoded = model.decoder(x)  # (1, D_POSE, N)
        else:
            # For regular decoder
            decoded = model(x)  # (1, D_POSE, N)
        motion_np = decoded.squeeze(0).cpu().numpy().T  # (N, D_POSE)

    return motion_np


def encode_motion_to_tokens(
    features, model, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Encode motion to discrete tokens using RVQVAE.

    Args:
        features: (N, D_POSE) numpy array
        model: RVQVAE model
        device: device to run on

    Returns:
        tokens: List of (n,) numpy arrays, one per quantization layer
    """
    if not isinstance(model, RVQVAE):
        raise ValueError("Model must be RVQVAE for token encoding")

    model.eval()
    model.to(device)

    # Prepare input: (1, D_POSE, N)
    N, D = features.shape
    x = torch.from_numpy(features.T).unsqueeze(0).float().to(device)  # (1, D, N)

    with torch.no_grad():
        tokens = model.encode_to_tokens(x)  # List of (1, n) tensors
        tokens_np = [t.squeeze(0).cpu().numpy() for t in tokens]  # List of (n,) arrays

    return tokens_np


def load_rvq_vae(
    checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load trained RVQVAE model from checkpoint.

    Args:
        checkpoint_path: Path to RVQVAE checkpoint (.pth file)
        device: Device to load model on

    Returns:
        model: RVQVAE model
        config: Dictionary with model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        print("Warning: No config found in checkpoint, using defaults")
        config = {}

    # Get model parameters from config or use defaults
    input_dim = config.get("input_dim", config.get("D_POSE", 256))
    output_dim = config.get("output_dim", config.get("D_POSE", 256))
    latent_dim = config.get("latent_dim", 256)
    downsampling_ratio = config.get("downsampling_ratio", 4)
    num_layers = config.get("num_layers", 4)
    hidden_dim = config.get("hidden_dim", 512)
    num_quantizers = config.get("num_quantizers", 6)
    num_embeddings = config.get("num_embeddings", 512)
    commitment_cost = config.get("commitment_cost", 1.0)
    decay = config.get("decay", 0.99)
    quantization_dropout = config.get("quantization_dropout", 0.2)

    # Create model
    model = RVQVAE(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_quantizers=num_quantizers,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        decay=decay,
        quantization_dropout=quantization_dropout,
    )

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, config


from tqdm import tqdm


def tokenize_all_motions(
    vq_model: RVQVAE,
    motion_dir_: str,
    metadata_dir_: str,
    mean: np.ndarray,
    std: np.ndarray,
    device: str = "cuda",
    cache_path: Optional[str] = None,
) -> dict:
    """
    Tokenize all motion files using the RVQ-VAE.

    Returns:
        Dictionary mapping sentence_id -> {
            'tokens': np.array (num_quantizers, n),
            'sentence': str,
            'gloss': str
        }
    """
    # Check cache first
    if cache_path and os.path.exists(cache_path):
        print(f"Loading tokenized data from cache: {cache_path}")
        return np.load(cache_path, allow_pickle=True).item()

    motion_dir = Path(motion_dir_)
    metadata_dir = Path(metadata_dir_)

    vq_model.eval()
    vq_model.to(device)

    token_data = {}
    motion_files = list(motion_dir.glob("*.npy"))

    print(f"Tokenizing {len(motion_files)} motion files...")

    for motion_file in tqdm(motion_files, desc="Tokenizing"):
        sid = motion_file.stem
        metadata_path = metadata_dir / sid / "metadata.txt"

        if not metadata_path.exists():
            continue

        try:
            # Load motion
            motion = np.load(motion_file).astype(np.float32)

            # Normalize
            motion = (motion - mean) / std

            # Tokenize
            tokens = encode_motion_to_tokens(motion, vq_model, device)
            tokens = np.stack(tokens, axis=0)  # (num_quantizers, n)

            # Load metadata
            metadata = parse_metadata(str(metadata_path))

            token_data[sid] = {
                "tokens": tokens,
                "sentence": metadata["sentence"],
                "gloss": metadata["gloss"],
            }
        except Exception as e:
            print(f"Warning: Failed to process {sid}: {e}")
            continue

    # Save cache
    if cache_path:
        cache_dir = Path(cache_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, token_data, allow_pickle=True)  # ty:ignore[invalid-argument-type]
        print(f"Saved tokenized data to cache: {cache_path}")

    return token_data
