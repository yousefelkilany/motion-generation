#!/usr/bin/env python3
"""
Masked Transformer with CROSS-ATTENTION for Sign Language Motion Generation.
Based on MoMask paper: Generative Masked Modeling of 3D Human Motions.

Architecture per layer:
    Self-Attention (motion → motion) → Cross-Attention (motion → text) → FFN

This architecture forces the model to attend to text via dedicated cross-attention
layers, preventing the "text conditioning collapse" that occurs with concatenation.
"""

import math
from collections.abc import Callable
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_encoder import CLIPTextEncoder, TextProjector


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    Cosine noise schedule for masking.
    Returns mask probability at timestep t (0 to 1).
    """
    return torch.cos(t * math.pi / 2)


def uniform(shape, device):
    """Sample uniform random values in [0, 1)."""
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def top_k(logits: torch.Tensor, thres: float = 0.9, dim: int = -1) -> torch.Tensor:
    """Filter logits to keep only top-k values."""
    k = max(1, int((1 - thres) * logits.shape[dim]))
    val, ind = logits.topk(k, dim=dim)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(dim, ind, val)
    return probs


def gumbel_sample(
    logits: torch.Tensor, temperature: float = 1.0, dim: int = -1
) -> torch.Tensor:
    """Gumbel-softmax sampling."""
    return ((logits / max(temperature, 1e-10)) + gumbel_noise(logits)).argmax(dim=dim)


def gumbel_noise(t: torch.Tensor) -> torch.Tensor:
    """Generate Gumbel noise."""
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -torch.log(-torch.log(noise.clamp(1e-20)).clamp(1e-20))


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Convert sequence lengths to boolean mask.

    Args:
        lengths: (B,) tensor of sequence lengths
        max_len: Maximum sequence length

    Returns:
        mask: (B, max_len) boolean tensor (True = valid, False = padding)
    """
    device = lengths.device
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch, d_model)
        """
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class CrossAttentionBlock(nn.Module):
    """
    Transformer block with Self-Attention + Cross-Attention.

    Architecture:
        1. Self-Attention: motion tokens attend to motion tokens
        2. Cross-Attention: motion tokens attend to text tokens (FORCED!)
        3. FFN: feedforward transformation

    This ensures text conditioning cannot be ignored - the model MUST look at text.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        # Self-attention (motion → motion) with pre-norm
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)

        # Cross-attention (motion → text) with pre-norm
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_dropout = nn.Dropout(dropout)

        # FFN with pre-norm
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        motion: torch.Tensor,
        text: torch.Tensor,
        motion_key_padding_mask: Optional[torch.Tensor] = None,
        text_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            motion: (seq_len, B, d_model) motion token embeddings
            text: (text_len, B, d_model) text token embeddings
            motion_key_padding_mask: (B, seq_len) True = padding position
            text_key_padding_mask: (B, text_len) True = padding position

        Returns:
            motion: (seq_len, B, d_model) updated motion embeddings
        """
        # 1. Self-attention: motion attends to motion
        residual = motion
        motion_norm = self.self_attn_norm(motion)
        self_attn_out, _ = self.self_attn(
            query=motion_norm,
            key=motion_norm,
            value=motion_norm,
            key_padding_mask=motion_key_padding_mask,
        )
        motion = residual + self.self_attn_dropout(self_attn_out)

        # 2. Cross-attention: motion attends to text (CRITICAL for conditioning!)
        residual = motion
        motion_norm = self.cross_attn_norm(motion)
        cross_attn_out, _ = self.cross_attn(
            query=motion_norm,  # Motion queries
            key=text,  # Text keys
            value=text,  # Text values
            key_padding_mask=text_key_padding_mask,
        )
        motion = residual + self.cross_attn_dropout(cross_attn_out)

        # 3. FFN
        residual = motion
        motion_norm = self.ffn_norm(motion)
        motion = residual + self.ffn(motion_norm)

        return motion


class InputProcess(nn.Module):
    """Process token embeddings for transformer input."""

    def __init__(self, code_dim: int, latent_dim: int):
        super().__init__()
        self.embedding = nn.Linear(code_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, code_dim)
        Returns:
            (seq_len, B, latent_dim)
        """
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, B, latent_dim)
        return x


class OutputProcess(nn.Module):
    """Process transformer output to logits."""

    def __init__(self, latent_dim: int, num_tokens: int):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        self.output = nn.Linear(latent_dim, num_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, B, latent_dim)
        Returns:
            (B, num_tokens, seq_len) logits
        """
        x = self.dense(x)
        x = F.gelu(x)
        x = self.ln(x)
        x = self.output(x)  # (seq_len, B, num_tokens)
        x = x.permute(1, 2, 0)  # (B, num_tokens, seq_len)
        return x


class MaskTransformer(nn.Module):
    """
    Masked Transformer with Cross-Attention for motion generation (MoMask-style).

    Takes motion tokens (from RVQ-VAE) and text conditioning,
    learns to predict masked tokens using BERT-style masking.

    Key Architecture Feature:
        Uses CROSS-ATTENTION for text conditioning instead of concatenation.
        This forces the model to attend to text in every layer, preventing
        the common failure mode where the model ignores text.

    Architecture per layer:
        Self-Attention(motion) → Cross-Attention(motion→text) → FFN
    """

    def __init__(
        self,
        num_tokens: int,
        code_dim: int,
        latent_dim: int = 384,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 6,
        dropout: float = 0.1,
        clip_dim: int = 512,
        clip_version: str = "ViT-B/32",
        cond_drop_prob: float = 0.1,
        device: str = "cuda",
        max_seq_len: int = 600,
    ):
        """
        Args:
            num_tokens: Size of motion token vocabulary (codebook size)
            code_dim: Dimension of motion token embeddings
            latent_dim: Transformer hidden dimension
            ff_size: Feedforward dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            clip_dim: CLIP embedding dimension
            clip_version: CLIP model version
            cond_drop_prob: Probability of dropping text condition (for CFG)
            device: Device to run on
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.num_tokens: int = num_tokens
        self.code_dim: int = code_dim
        self.latent_dim: int = latent_dim
        self.cond_drop_prob: float = cond_drop_prob

        # Special tokens
        self.mask_id: int = num_tokens  # MASK token ID
        self.pad_id: int = num_tokens + 1  # PAD token ID

        # Token embedding (+2 for MASK and PAD)
        self.token_emb = nn.Embedding(num_tokens + 2, code_dim)

        # Input/output processing
        self.input_process = InputProcess(code_dim, latent_dim)
        self.output_process = OutputProcess(latent_dim, num_tokens)

        # Positional encoding for motion tokens
        self.pos_encoder = PositionalEncoding(latent_dim, dropout, max_seq_len)

        # Text encoder (CLIP) - frozen
        self.text_encoder = CLIPTextEncoder(
            clip_version=clip_version, device=device, freeze=True
        )

        # Text projection to match latent_dim
        self.text_proj = TextProjector(clip_dim, latent_dim, dropout)

        # Cross-attention transformer layers
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(latent_dim)

        # Noise schedule for masking
        self.noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule

        # Initialize weights
        self.apply(self._init_weights)

        print("MaskTransformer (Cross-Attention) initialized:")
        print(f"  - Vocab size: {num_tokens} (+2 special tokens)")
        print(f"  - Latent dim: {latent_dim}")
        print(f"  - Layers: {num_layers}, Heads: {num_heads}")
        print("  - Architecture: Self-Attn → Cross-Attn → FFN (per layer)")
        print(f"  - Condition drop prob: {cond_drop_prob}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def mask_cond(self, cond: torch.Tensor, force_mask: bool = False) -> torch.Tensor:
        """
        Apply condition dropout for classifier-free guidance.

        Args:
            cond: (B, seq_len, dim) text condition
            force_mask: If True, always mask (for CFG null condition)
        """
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0:
            bs = cond.shape[0]
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_drop_prob
            )
            mask = mask.view(bs, 1, 1)
            return cond * (1 - mask)
        else:
            return cond

    def encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text using CLIP.

        Returns:
            cond_emb: (B, seq_len, latent_dim) projected text embeddings
            cond_mask: (B, seq_len) attention mask (1 = valid, 0 = padding)
        """
        # Get token-level embeddings from CLIP
        text_emb, text_mask = self.text_encoder.encode_text_tokens(texts)

        # Project to latent dimension
        cond_emb = self.text_proj(text_emb)  # (B, seq_len, latent_dim)

        return cond_emb, text_mask

    def forward_transformer(
        self,
        motion_ids: torch.Tensor,
        cond_emb: torch.Tensor,
        cond_padding_mask: torch.Tensor,
        motion_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention transformer.

        Args:
            motion_ids: (B, seq_len) motion token IDs
            cond_emb: (B, text_len, latent_dim) text embeddings
            cond_padding_mask: (B, text_len) True = padding
            motion_padding_mask: (B, seq_len) True = padding

        Returns:
            logits: (B, num_tokens, seq_len)
        """
        B, seq_len = motion_ids.shape

        # Embed motion tokens
        x = self.token_emb(motion_ids)  # (B, seq_len, code_dim)
        x = self.input_process(x)  # (seq_len, B, latent_dim)

        # Add positional encoding to motion
        x = self.pos_encoder(x)

        # Prepare text for cross-attention (seq_first format)
        # Note: CLIP embeddings already contain positional information
        text = cond_emb.permute(1, 0, 2)  # (text_len, B, latent_dim)

        # Apply cross-attention layers
        for layer in self.layers:
            x = layer(
                motion=x,
                text=text,
                motion_key_padding_mask=motion_padding_mask,
                text_key_padding_mask=cond_padding_mask,
            )

        # Final normalization
        x = self.final_norm(x)

        # Project to logits
        logits = self.output_process(x)  # (B, num_tokens, seq_len)

        return logits

    def forward(
        self,
        motion_ids: torch.Tensor,
        texts: List[str],
        m_lens: torch.Tensor,
        full_mask_prob: float = 0.15,
        label_smoothing: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Training forward pass with random masking.

        Args:
            motion_ids: (B, seq_len) ground truth motion token IDs
            texts: List of text strings
            m_lens: (B,) sequence lengths
            full_mask_prob: Probability of masking ALL tokens (forces text dependence)
            label_smoothing: Label smoothing factor to prevent overconfidence (0.0-0.2)

        Returns:
            loss: Cross-entropy loss
            pred_ids: (B, seq_len) predicted token IDs
            accuracy: Prediction accuracy on masked tokens
        """
        B, seq_len = motion_ids.shape
        device = motion_ids.device

        # Create padding mask (True = padding)
        non_pad_mask = lengths_to_mask(m_lens, seq_len)  # (B, seq_len)
        padding_mask = ~non_pad_mask

        # Replace padding positions with PAD token
        motion_ids = torch.where(non_pad_mask, motion_ids, self.pad_id)

        # Encode text
        with torch.no_grad():
            cond_emb, cond_att_mask = self.encode_text(texts)

        cond_padding_mask = cond_att_mask == 0  # True = padding

        # Apply condition dropout (for CFG training)
        cond_emb = self.mask_cond(cond_emb)

        # Random masking with cosine schedule
        rand_time = uniform((B,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)  # (B,)
        num_tokens_to_mask = (seq_len * rand_mask_probs).round().clamp(min=1)

        # CRITICAL: With some probability, mask ALL tokens
        # This forces the model to rely on text conditioning rather than motion context
        full_mask_samples = torch.bernoulli(
            torch.full((B,), full_mask_prob, device=device)
        ).bool()

        # For full-mask samples, set num_tokens_to_mask to sequence length
        num_tokens_to_mask = torch.where(
            full_mask_samples,
            m_lens.float(),  # Mask all valid tokens
            num_tokens_to_mask,
        )

        # Create random mask
        batch_randperm = torch.rand((B, seq_len), device=device).argsort(dim=-1)
        mask = batch_randperm < num_tokens_to_mask.unsqueeze(-1)

        # Only mask non-padding positions
        mask = mask & non_pad_mask

        # Create training target (masked positions are target, others are ignored)
        labels = torch.where(mask, motion_ids, self.mask_id)

        # Create input (replace masked positions with MASK token)
        x_ids = motion_ids.clone()

        # BERT-style masking:
        # 80% -> MASK token
        # 10% -> random token
        # 10% -> keep original

        # 10% random replacement
        mask_random = (
            torch.bernoulli(torch.full((B, seq_len), 0.1, device=device)).bool() & mask
        )
        rand_tokens = torch.randint(0, self.num_tokens, (B, seq_len), device=device)
        x_ids = torch.where(mask_random, rand_tokens, x_ids)

        # 80% MASK token (on non-random masked positions)
        mask_token = (
            torch.bernoulli(torch.full((B, seq_len), 0.8, device=device)).bool()
            & mask
            & ~mask_random
        )
        x_ids = torch.where(mask_token, self.mask_id, x_ids)

        # Forward through transformer
        logits = self.forward_transformer(
            x_ids, cond_emb, cond_padding_mask, padding_mask
        )

        # Compute loss only on masked positions
        logits_t = logits.permute(0, 2, 1).contiguous()  # (B, seq_len, num_tokens)

        loss = F.cross_entropy(
            logits_t.reshape(-1, self.num_tokens),
            labels.reshape(-1),
            ignore_index=self.mask_id,
            reduction="mean",
            label_smoothing=label_smoothing,
        )

        # Get predictions
        pred_ids = logits.argmax(dim=1)  # (B, seq_len)

        # Compute accuracy on masked positions
        correct = (pred_ids == motion_ids) & mask
        accuracy = correct.sum().float() / mask.sum().clamp(min=1)

        return loss, pred_ids, accuracy.item()

    def forward_with_cfg(
        self,
        motion_ids: torch.Tensor,
        cond_emb: torch.Tensor,
        cond_padding_mask: torch.Tensor,
        motion_padding_mask: torch.Tensor,
        cond_scale: float = 3.0,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.

        Args:
            motion_ids: (B, seq_len) motion token IDs with MASK tokens
            cond_emb: (B, text_len, latent_dim) text embeddings
            cond_padding_mask: (B, text_len) True = padding
            motion_padding_mask: (B, seq_len) True = padding
            cond_scale: Guidance scale

        Returns:
            logits: (B, num_tokens, seq_len) scaled logits
        """
        # Forward with condition
        logits_cond = self.forward_transformer(
            motion_ids, cond_emb, cond_padding_mask, motion_padding_mask
        )

        # Forward without condition (null condition)
        logits_uncond = self.forward_transformer(
            motion_ids,
            self.mask_cond(cond_emb, force_mask=True),
            cond_padding_mask,
            motion_padding_mask,
        )

        # CFG: logits = uncond + scale * (cond - uncond)
        logits = logits_uncond + cond_scale * (logits_cond - logits_uncond)

        return logits

    @torch.no_grad()
    def generate(
        self,
        texts: List[str],
        m_lens: torch.Tensor,
        timesteps: int = 10,
        cond_scale: float = 4.0,
        temperature: float = 1.0,
        topk_filter_thres: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate motion tokens from text using iterative demasking.

        Args:
            texts: List of text prompts
            m_lens: (B,) target motion lengths
            timesteps: Number of demasking steps
            cond_scale: Classifier-free guidance scale
            temperature: Sampling temperature
            topk_filter_thres: Top-k filtering threshold

        Returns:
            ids: (B, max_len) generated motion token IDs
        """
        self.eval()

        B = len(texts)
        device = next(self.parameters()).device
        max_len = int(m_lens.max().item())

        # Create padding mask
        non_pad_mask = lengths_to_mask(m_lens, max_len)
        padding_mask = ~non_pad_mask

        # Encode text
        cond_emb, cond_att_mask = self.encode_text(texts)
        cond_padding_mask = cond_att_mask == 0

        # Start with all MASK tokens
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.0)

        # Iterative demasking
        for step in range(timesteps):
            # Get current mask probability based on timestep
            t = torch.tensor(step / timesteps, device=device)
            mask_prob = self.noise_schedule(t)

            # Number of tokens to keep masked
            num_masked = torch.round(mask_prob * m_lens.float()).clamp(min=1).long()

            # Select tokens with lowest scores to remain masked
            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = ranks < num_masked.unsqueeze(-1)
            ids = torch.where(is_mask, self.mask_id, ids)

            # Forward with CFG
            logits = self.forward_with_cfg(
                ids, cond_emb, cond_padding_mask, padding_mask, cond_scale
            )  # (B, num_tokens, seq_len)

            logits = logits.permute(0, 2, 1)  # (B, seq_len, num_tokens)

            # Apply top-k filtering
            filtered_logits = top_k(logits, topk_filter_thres)

            # Sample
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            pred_ids = torch.multinomial(probs.view(-1, self.num_tokens), 1).view(
                B, max_len
            )

            # Update IDs at masked positions
            ids = torch.where(is_mask & non_pad_mask, pred_ids, ids)

            # Update scores (confidence for each token)
            probs_no_temp = F.softmax(logits, dim=-1)
            scores = probs_no_temp.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)
            scores = torch.where(
                ~is_mask, 1e5, scores
            )  # Don't re-mask already revealed tokens

        # Replace PAD tokens with -1 for clarity
        ids = torch.where(padding_mask, -1, ids)

        return ids

    def parameters_wo_clip(self):
        """Get parameters excluding CLIP (for optimizer)."""
        return [p for n, p in self.named_parameters() if "text_encoder" not in n]
