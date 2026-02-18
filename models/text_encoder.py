"""
CLIP Text Encoder for conditioning the Masked Transformer.
Uses OpenAI's CLIP model to encode text prompts into embeddings.
"""

from config import DEVICE, TRANSFORMER_CONFIG

from typing import List, Tuple

import torch
import torch.nn as nn


class CLIPTextEncoder(nn.Module):
    """
    CLIP-based text encoder for motion generation conditioning.

    Uses the text encoder from CLIP to get text embeddings.
    Can output either:
    - Global embedding (single vector per sentence)
    - Token embeddings (sequence of embeddings for each token)
    """

    def __init__(
        self,
        clip_version: str = "ViT-B/32",
        device: str = "cuda",
        max_text_length: int = 77,
        freeze: bool = True,
    ):
        """
        Args:
            clip_version: CLIP model version (e.g., "ViT-B/32", "ViT-L/14")
            device: Device to load model on
            max_text_length: Maximum text length for CLIP tokenizer
            freeze: Whether to freeze CLIP weights
        """
        super().__init__()

        self.device: str = device
        self.max_text_length: int = max_text_length
        self.freeze: bool = freeze

        # Import CLIP
        try:
            import clip
        except ImportError:
            raise ImportError(
                "Please install CLIP: pip install git+https://github.com/openai/CLIP.git"
            )

        # Load CLIP model
        print(f"Loading CLIP model: {clip_version}")
        self.clip_model, self.preprocess = clip.load(clip_version, device=device)

        # Ensure CLIP is in float32 to avoid dtype mismatch errors
        self.clip_model = self.clip_model.float()

        # Get embedding dimension
        self.embed_dim = self.clip_model.text_projection.shape[1]
        print(f"CLIP text embedding dimension: {self.embed_dim}")

        # Freeze CLIP if specified
        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
            print("CLIP weights frozen")

        # Store clip module for tokenization
        self.clip = clip  # ty:ignore[unresolved-attribute]

    def tokenize(self, texts: str | List[str]) -> torch.Tensor:
        """
        Tokenize text using CLIP tokenizer.

        Args:
            texts: List of text strings

        Returns:
            tokens: (B, max_text_length) tensor of token IDs
        """
        tokens = self.clip.tokenize(texts, truncate=True).to(self.device)
        return tokens

    def encode_text(self, texts: str | List[str]) -> torch.Tensor:
        """
        Encode text to global CLIP embeddings.

        Args:
            texts: List of text strings

        Returns:
            embeddings: (B, embed_dim) tensor
        """
        tokens = self.tokenize(texts)

        with torch.no_grad() if self.freeze else torch.enable_grad():
            embeddings = self.clip_model.encode_text(tokens)

        # Normalize embeddings (CLIP does this)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.float()

    def encode_text_tokens(
        self, texts: str | List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text to token-level embeddings (before final projection).
        Useful for cross-attention conditioning.

        Args:
            texts: List of text strings

        Returns:
            embeddings: (B, seq_len, embed_dim) tensor
            attention_mask: (B, seq_len) tensor (1 = valid, 0 = padding)
        """
        tokens = self.tokenize(texts)  # (B, 77)

        with torch.no_grad() if self.freeze else torch.enable_grad():
            # Get token embeddings before projection
            x = self.clip_model.token_embedding(tokens)  # (B, 77, 512)
            x = x + self.clip_model.positional_embedding
            x = x.permute(1, 0, 2)  # (77, B, 512)
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)  # (B, 77, 512)
            x = self.clip_model.ln_final(x)

        # Create attention mask (1 for valid tokens, 0 for padding)
        # CLIP uses <|endoftext|> as EOS token (ID = 49407)
        attention_mask = (tokens != 0).float()  # Padding tokens are 0

        return x.float(), attention_mask

    def forward(
        self, texts: str | List[str], return_tokens: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            texts: List of text strings
            return_tokens: If True, return token-level embeddings

        Returns:
            If return_tokens=False: (B, embed_dim) global embeddings
            If return_tokens=True: (B, seq_len, embed_dim) token embeddings
        """
        if return_tokens:
            embeddings, mask = self.encode_text_tokens(texts)
            return embeddings, mask
        else:
            return self.encode_text(texts)


class TextProjector(nn.Module):
    """
    Projects text embeddings to transformer latent dimension.
    """

    def __init__(self, text_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(text_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


if __name__ == "__main__":
    # Initialize CLIP Text Encoder
    # Using a common CLIP model like "ViT-B/32" for text embedding
    clip_version = "ViT-B/32"
    clip_text_encoder = CLIPTextEncoder(clip_version=clip_version, device=DEVICE)

    # Initialize Text Projector
    # The output dimension should match the transformer's latent dimension
    # (Assuming transformer's latent_dim is 384 from TRANSFORMER_CONFIG in next section)
    text_projector = TextProjector(
        text_dim=512, latent_dim=int(TRANSFORMER_CONFIG["latent_dim"])
    ).to(DEVICE)

    # Example: Encode a text prompt
    example_text_prompt = "Hello, How are you?"
    print(f'Encoding text prompt: "{example_text_prompt}"')

    with torch.no_grad():
        text_embedding = clip_text_encoder.encode_text(example_text_prompt)
        projected_text_embedding = text_projector(text_embedding)

    print(f"Original CLIP embedding shape: {text_embedding.shape}")
    print(f"Projected text embedding shape: {projected_text_embedding.shape}")
