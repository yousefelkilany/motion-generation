"""
Residual Transformer for hierarchical motion token generation.

Predicts residual layer tokens (layers 1-5) given previous layers.
Based on MoMask paper: Generative Masked Modeling of 3D Human Motions.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rvq_vae import RVQVAE
from models.text_encoder import CLIPTextEncoder, TextProjector
from models.transformer import (
    InputProcess,
    OutputProcess,
    PositionalEncoding,
    lengths_to_mask,
)


class ResidualTransformer(nn.Module):
    """
    Residual Transformer for predicting residual layer tokens.

    Given tokens from layers 0:j-1, predicts tokens for layer j.
    Uses layer indicator j for conditioning.
    """

    def __init__(
        self,
        num_tokens: int,
        code_dim: int,
        num_quantizers: int,
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
        share_weight: bool = True,
    ):
        """
        Args:
            num_tokens: Size of motion token vocabulary (codebook size)
            code_dim: Dimension of motion token embeddings
            num_quantizers: Total number of quantizer layers (including base)
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
            share_weight: Whether to share weights between layers (MoMask style)
        """
        super().__init__()

        self.num_tokens: int = num_tokens
        self.code_dim: int = code_dim
        self.num_quantizers: int = num_quantizers
        self.latent_dim: int = latent_dim
        self.cond_drop_prob: float = cond_drop_prob
        self.share_weight: bool = share_weight

        # Special tokens
        self.pad_id: int = num_tokens  # PAD token ID

        # Token embeddings for each residual layer (1 to num_quantizers-1)
        # Layer 0 uses base transformer, so we start from layer 1
        if share_weight:
            # Shared embedding across all residual layers
            self.token_emb = nn.Embedding(num_tokens + 1, code_dim)  # +1 for PAD
        else:
            # Separate embedding for each residual layer
            self.token_emb = nn.ModuleList(
                [
                    nn.Embedding(num_tokens + 1, code_dim)
                    for _ in range(num_quantizers - 1)
                ]
            )

        # Layer indicator embedding (which residual layer we're predicting)
        self.layer_emb = nn.Embedding(num_quantizers - 1, latent_dim)

        # Input/output processing
        self.input_process = InputProcess(code_dim, latent_dim)

        # Output layers: one per residual layer (or shared if share_weight=True)
        if share_weight:
            self.output_process = OutputProcess(latent_dim, num_tokens)
        else:
            self.output_process = nn.ModuleList(
                [
                    OutputProcess(latent_dim, num_tokens)
                    for _ in range(num_quantizers - 1)
                ]
            )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(latent_dim, dropout, max_seq_len)

        # Text encoder (CLIP) - shared with base transformer
        self.text_encoder = CLIPTextEncoder(
            clip_version=clip_version, device=device, freeze=True
        )

        # Text projection
        self.text_proj = TextProjector(clip_dim, latent_dim, dropout)

        # Transformer encoder (in-context conditioning)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Initialize weights
        self.apply(self._init_weights)

        print("ResidualTransformer initialized:")
        print(f"  - Vocab size: {num_tokens} (+1 special token)")
        print(f"  - Code dim: {code_dim}")
        print(f"  - Latent dim: {latent_dim}")
        print(f"  - Layers: {num_layers}, Heads: {num_heads}")
        print(f"  - Residual layers: {num_quantizers - 1}")
        print(f"  - Share weights: {share_weight}")

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
            cond_mask: (B, seq_len) attention mask
        """
        # Get token-level embeddings
        text_emb, text_mask = self.text_encoder.encode_text_tokens(texts)

        # Project to latent dimension
        cond_emb = self.text_proj(text_emb)  # (B, seq_len, latent_dim)

        return cond_emb, text_mask

    def embed_previous_layers(
        self, prev_layer_tokens: List[torch.Tensor], vq_model: RVQVAE
    ) -> torch.Tensor:
        """
        Embed tokens from previous layers by summing their codebook embeddings.

        Args:
            prev_layer_tokens: List of (B, seq_len) token tensors for layers 0:j-1
            vq_model: RVQ-VAE model to get codebook embeddings

        Returns:
            (B, seq_len, code_dim) summed embeddings
        """
        B, seq_len = prev_layer_tokens[0].shape
        device = prev_layer_tokens[0].device

        # Sum embeddings from all previous layers
        # Each layer's tokens index into its codebook
        summed_emb = torch.zeros(B, seq_len, self.code_dim, device=device)

        for layer_idx, tokens in enumerate(prev_layer_tokens):
            # Get codebook for this layer
            # (code_dim, num_embeddings)
            codebook = vq_model.rvq.quantizers[layer_idx].embedding

            # Lookup embeddings: (B, seq_len) -> (B, seq_len, code_dim)
            flat_tokens = tokens.reshape(
                -1
            )  # (B*seq_len,) - use reshape instead of view for non-contiguous tensors
            # Handle padding: replace pad_id with 0 index
            flat_tokens = torch.clamp(flat_tokens, min=0, max=self.num_tokens - 1)
            layer_emb = codebook[:, flat_tokens].t()  # (B*seq_len, code_dim)
            layer_emb = layer_emb.reshape(
                B, seq_len, self.code_dim
            )  # Use reshape instead of view

            # Sum with previous layers
            summed_emb = summed_emb + layer_emb

        return summed_emb

    def forward_transformer(
        self,
        prev_layer_emb: torch.Tensor,
        layer_idx: int,
        cond_emb: torch.Tensor,
        cond_padding_mask: torch.Tensor,
        motion_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through transformer.

        Args:
            prev_layer_emb: (B, seq_len, code_dim) embeddings from previous layers
            layer_idx: Which residual layer we're predicting (1 to num_quantizers-1)
            cond_emb: (B, text_len, latent_dim) text embeddings
            cond_padding_mask: (B, text_len) True = padding
            motion_padding_mask: (B, seq_len) True = padding

        Returns:
            logits: (B, num_tokens, seq_len)
        """
        B, seq_len = prev_layer_emb.shape[:2]
        text_len = cond_emb.shape[1]

        # Embed previous layers
        x = self.input_process(prev_layer_emb)  # (seq_len, B, latent_dim)

        # Add layer indicator embedding
        layer_indicator = self.layer_emb(
            torch.tensor(layer_idx - 1, device=x.device)
        )  # (latent_dim,)
        # Broadcast to all positions
        x = x + layer_indicator.unsqueeze(0).unsqueeze(1)  # (seq_len, B, latent_dim)

        # Prepare text condition
        cond = cond_emb.permute(1, 0, 2)  # (text_len, B, latent_dim)

        # Add positional encoding
        # Combined positional encoding to avoid collision between text and motion
        xseq = torch.cat([cond, x], dim=0)  # (text_len + seq_len, B, latent_dim)
        xseq = self.pos_encoder(xseq)

        # Create combined padding mask
        padding_mask = torch.cat(
            [cond_padding_mask, motion_padding_mask], dim=1
        )  # (B, text_len + seq_len)

        # Transformer forward
        output = self.transformer(xseq, src_key_padding_mask=padding_mask)

        # Take only motion part
        output = output[text_len:]  # (seq_len, B, latent_dim)

        # Project to logits
        if self.share_weight:
            logits = self.output_process(output)  # (B, num_tokens, seq_len)
        else:
            # (B, num_tokens, seq_len)
            self.output_process: nn.ModuleList
            logits = self.output_process[layer_idx - 1](output)

        return logits

    def forward(
        self,
        prev_layer_tokens: List[torch.Tensor],
        target_tokens: torch.Tensor,
        layer_idx: int,
        texts: List[str],
        m_lens: torch.Tensor,
        vq_model: RVQVAE,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Training forward pass.

        Args:
            prev_layer_tokens: List of (B, seq_len) token tensors for layers 0:j-1
            target_tokens: (B, seq_len) ground truth token IDs for layer j
            layer_idx: Which residual layer we're predicting (1 to num_quantizers-1)
            texts: List of text strings
            m_lens: (B,) sequence lengths
            vq_model: RVQ-VAE model for codebook access

        Returns:
            loss: Cross-entropy loss
            pred_ids: (B, seq_len) predicted token IDs
            accuracy: Prediction accuracy
        """
        B, seq_len = target_tokens.shape

        # Create padding mask (True = padding)
        non_pad_mask = lengths_to_mask(m_lens, seq_len)  # (B, seq_len)
        padding_mask = ~non_pad_mask

        # Replace padding positions with PAD token
        target_tokens = torch.where(non_pad_mask, target_tokens, self.pad_id)

        # Embed previous layers
        prev_layer_emb = self.embed_previous_layers(prev_layer_tokens, vq_model)

        # Encode text
        with torch.no_grad():
            cond_emb, cond_att_mask = self.encode_text(texts)

        cond_padding_mask = cond_att_mask == 0  # True = padding

        # Apply condition dropout
        cond_emb = self.mask_cond(cond_emb)

        # Forward through transformer
        logits = self.forward_transformer(
            prev_layer_emb, layer_idx, cond_emb, cond_padding_mask, padding_mask
        )

        # Compute loss
        logits_t = logits.permute(0, 2, 1).contiguous()  # (B, seq_len, num_tokens)

        loss = F.cross_entropy(
            logits_t.reshape(-1, self.num_tokens),
            target_tokens.reshape(-1),
            ignore_index=self.pad_id,
            reduction="mean",
        )

        # Get predictions
        pred_ids = logits.argmax(dim=1)  # (B, seq_len)

        # Compute accuracy on non-padding positions
        correct = (pred_ids == target_tokens) & non_pad_mask
        accuracy = correct.sum().float() / non_pad_mask.sum().clamp(min=1)

        return loss, pred_ids, accuracy.item()

    def forward_with_cfg(
        self,
        prev_layer_emb: torch.Tensor,
        layer_idx: int,
        cond_emb: torch.Tensor,
        cond_padding_mask: torch.Tensor,
        motion_padding_mask: torch.Tensor,
        cond_scale: float = 3.0,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.

        Args:
            prev_layer_emb: (B, seq_len, code_dim) embeddings from previous layers
            layer_idx: Which residual layer we're predicting
            cond_emb: (B, text_len, latent_dim) text embeddings
            cond_padding_mask: (B, text_len) True = padding
            motion_padding_mask: (B, seq_len) True = padding
            cond_scale: Guidance scale

        Returns:
            logits: (B, num_tokens, seq_len) scaled logits
        """
        # Forward with condition
        logits_cond = self.forward_transformer(
            prev_layer_emb, layer_idx, cond_emb, cond_padding_mask, motion_padding_mask
        )

        # Forward without condition (null condition)
        logits_uncond = self.forward_transformer(
            prev_layer_emb,
            layer_idx,
            self.mask_cond(cond_emb, force_mask=True),
            cond_padding_mask,
            motion_padding_mask,
        )

        # CFG: logits = uncond + scale * (cond - uncond)
        logits = logits_uncond + cond_scale * (logits_cond - logits_uncond)

        return logits

    @torch.no_grad()
    def generate_layer(
        self,
        prev_layer_tokens: List[torch.Tensor],
        layer_idx: int,
        texts: List[str],
        m_lens: torch.Tensor,
        vq_model: RVQVAE,
        cond_scale: float = 5.0,
        temperature: float = 1.0,
        topk_filter_thres: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate tokens for a specific residual layer.

        Args:
            prev_layer_tokens: List of (B, seq_len) token tensors for layers 0:j-1
            layer_idx: Which residual layer to generate (1 to num_quantizers-1)
            texts: List of text prompts
            m_lens: (B,) target motion lengths
            vq_model: RVQ-VAE model for codebook access
            cond_scale: Classifier-free guidance scale
            temperature: Sampling temperature
            topk_filter_thres: Top-k filtering threshold

        Returns:
            ids: (B, seq_len) generated token IDs for layer j
        """
        self.eval()

        B = len(texts)
        seq_len = prev_layer_tokens[0].shape[1]

        # Create padding mask
        non_pad_mask = lengths_to_mask(m_lens, seq_len)
        padding_mask = ~non_pad_mask

        # Embed previous layers
        prev_layer_emb = self.embed_previous_layers(prev_layer_tokens, vq_model)

        # Encode text
        cond_emb, cond_att_mask = self.encode_text(texts)
        cond_padding_mask = cond_att_mask == 0

        # Forward with CFG
        logits = self.forward_with_cfg(
            prev_layer_emb,
            layer_idx,
            cond_emb,
            cond_padding_mask,
            padding_mask,
            cond_scale,
        )  # (B, num_tokens, seq_len)

        logits = logits.permute(0, 2, 1)  # (B, seq_len, num_tokens)

        # Apply top-k filtering
        k = max(1, int((1 - topk_filter_thres) * logits.shape[-1]))
        topk_logits, topk_indices = logits.topk(k, dim=-1)
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(-1, topk_indices, topk_logits)

        # Sample
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        pred_ids = torch.multinomial(probs.view(-1, self.num_tokens), 1).view(
            B, seq_len
        )

        # Replace padding with PAD token
        pred_ids = torch.where(non_pad_mask, pred_ids, self.pad_id)

        return pred_ids

    def parameters_wo_clip(self):
        """Get parameters excluding CLIP (for optimizer)."""
        return [p for n, p in self.named_parameters() if "text_encoder" not in n]
