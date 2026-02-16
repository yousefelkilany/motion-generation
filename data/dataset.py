import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TokenizedMotionDataset(Dataset):
    """
    Dataset of pre-tokenized motion sequences with text captions.
    Returns all RVQ layers (base + residual layers).

    Text source options:
        - 'sentence': Natural English sentence only
        - 'gloss': Sign language gloss only (better motion alignment)
        - 'both': Concatenated "Sentence: {sentence} Signs: {gloss}" (recommended)
        - 'random': Randomly sample sentence or gloss each time (data augmentation)
    """

    # Format template for combined text (matches SignMotionDataset)
    COMBINED_FORMAT = "Sentence: {sentence} Signs: {gloss}"

    def __init__(
        self,
        token_data: dict,
        text_source: str = "both",
        max_token_len: int = 80,
        min_token_len: int = 6,
        return_all_layers: bool = True,
    ):
        """
        Args:
            token_data: Dictionary mapping sentence_id -> {
                'tokens': np.array (num_quantizers, n),
                'sentence': str,
                'gloss': str
            }
            text_source: 'sentence', 'gloss', 'both', or 'random'
            max_token_len: Maximum token sequence length
            min_token_len: Minimum token sequence length
            return_all_layers: If True, return all layers; else only base layer
        """
        if text_source not in ("sentence", "gloss", "both", "random"):
            raise ValueError(
                f"text_source must be 'sentence', 'gloss', 'both', or 'random', got '{text_source}'"
            )

        self.text_source = text_source
        self.max_token_len = max_token_len
        self.return_all_layers = return_all_layers

        # Filter by length
        self.data = []
        for sid, item in token_data.items():
            tokens = item["tokens"]
            # Handle both formats: (num_quantizers, n) or (n,)
            if len(tokens.shape) > 1:
                token_len = tokens.shape[1]
                num_quantizers = tokens.shape[0]
            else:
                token_len = len(tokens)
                num_quantizers = 1
                tokens = tokens[np.newaxis, :]  # Add quantizer dimension

            if min_token_len <= token_len <= max_token_len:
                self.data.append(
                    {
                        "id": sid,
                        "tokens": tokens,  # (num_quantizers, n) - all layers
                        "sentence": item["sentence"],
                        "gloss": item["gloss"],
                        "length": token_len,
                        "num_quantizers": num_quantizers,
                    }
                )

        print(
            f"TokenizedMotionDataset: {len(self.data)} samples (filtered from {len(token_data)})"
        )
        print(f"  Text source: '{text_source}'")
        if self.return_all_layers and len(self.data) > 0:
            print(f"  Returning all {self.data[0]['num_quantizers']} layers per sample")
        else:
            print("  Returning only base layer")

    def __len__(self):
        return len(self.data)

    def _get_text(self, item: dict) -> str:
        """Get text based on text_source setting."""
        if self.text_source == "sentence":
            return item["sentence"]
        elif self.text_source == "gloss":
            return item["gloss"]
        elif self.text_source == "both":
            return self.COMBINED_FORMAT.format(
                sentence=item["sentence"], gloss=item["gloss"]
            )
        elif self.text_source == "random":
            # Random augmentation: 50% sentence, 50% gloss
            if random.random() < 0.5:
                return item["sentence"]
            else:
                return item["gloss"]
        else:
            return item["sentence"]

    def __getitem__(self, index):
        item = self.data[index]

        tokens = item["tokens"].copy()  # (num_quantizers, n)
        text = self._get_text(item)
        length = item["length"]

        # Pad to max length for each layer
        if tokens.shape[1] < self.max_token_len:
            padding = np.zeros(
                (tokens.shape[0], self.max_token_len - tokens.shape[1]),
                dtype=tokens.dtype,
            )
            tokens = np.concatenate([tokens, padding], axis=1)
        else:
            tokens = tokens[:, : self.max_token_len]
            length = self.max_token_len

        # Convert to torch tensors
        tokens_torch = torch.from_numpy(
            tokens
        ).long()  # (num_quantizers, max_token_len)

        if self.return_all_layers:
            return text, tokens_torch, length
        else:
            # Return only base layer for backward compatibility
            return text, tokens_torch[0], length


def collate_fn(batch, return_all_layers=True):
    """Collate function for DataLoader."""
    texts, tokens, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)

    if return_all_layers and len(tokens[0].shape) > 1:
        # Multi-layer tokens: (num_quantizers, seq_len)
        # Stack to (B, num_quantizers, seq_len)
        tokens = torch.stack(tokens, dim=0)
    else:
        # Single layer tokens: (seq_len,)
        # Stack to (B, seq_len)
        tokens = torch.stack(tokens, dim=0)

    return list(texts), tokens, lengths


def parse_metadata(metadata_path: str) -> Dict[str, str]:
    """
    Parse metadata.txt file to extract SENTENCE and GLOSS.

    Args:
        metadata_path: Path to metadata.txt file

    Returns:
        Dictionary with 'sentence' and 'gloss' keys
    """
    result = {"sentence": "", "gloss": ""}

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("SENTENCE:"):
                    result["sentence"] = line[len("SENTENCE:") :].strip()
                elif line.startswith("GLOSS:"):
                    result["gloss"] = line[len("GLOSS:") :].strip()
    except Exception as e:
        print(f"Warning: Failed to parse {metadata_path}: {e}")

    return result
