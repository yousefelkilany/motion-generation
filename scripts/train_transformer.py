import random
from typing import Optional

import numpy as np
import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.residual_transformer import ResidualTransformer
from models.rvq_vae import RVQVAE
from models.transformer import MaskTransformer


def train_epoch(
    base_model: MaskTransformer,
    dataloader: DataLoader,
    base_optimizer: torch.optim.Optimizer,
    base_scheduler,
    device: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    residual_model: Optional[ResidualTransformer] = None,
    residual_optimizer: Optional[torch.optim.Optimizer] = None,
    residual_scheduler=None,
    vq_model: Optional[RVQVAE] = None,
    train_residual: bool = False,
    residual_prob: float = 0.5,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    gradient_accumulation_steps: int = 1,
    train_residual_only: bool = False,
    full_mask_prob: float = 0.15,
    label_smoothing: float = 0.1,
) -> dict:
    """
    Train for one epoch.

    Args:
        base_model: MaskTransformer for base layer
        dataloader: Training dataloader
        base_optimizer: Optimizer for base transformer
        base_scheduler: Scheduler for base transformer
        device: Device to run on
        epoch: Current epoch
        writer: TensorBoard writer
        residual_model: ResidualTransformer for residual layers (optional)
        residual_optimizer: Optimizer for residual transformer (optional)
        residual_scheduler: Scheduler for residual transformer (optional)
        vq_model: RVQ-VAE model (needed for residual transformer)
        train_residual: Whether to train residual transformer
        residual_prob: Probability of training residual transformer per batch
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP
        gradient_accumulation_steps: Number of steps to accumulate gradients
        train_residual_only: If True, only train residual transformer (skip base)
    """
    if not train_residual_only:
        base_model.train()
    else:
        base_model.eval()  # Keep base model frozen in residual-only mode

    if residual_model is not None:
        residual_model.train()

    total_loss = 0
    total_acc = 0
    total_res_loss = 0
    total_res_acc = 0
    num_batches = 0
    num_res_batches = 0

    # For gradient accumulation
    accumulated_loss = 0.0
    accumulated_res_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (texts, tokens, lengths) in enumerate(pbar):
        tokens = tokens.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        # Handle multi-layer tokens: (B, num_quantizers, seq_len) or (B, seq_len)
        if len(tokens.shape) == 3:
            # Multi-layer: extract base layer
            base_tokens = tokens[:, 0, :]  # (B, seq_len)
            all_layer_tokens = [tokens[:, i, :] for i in range(tokens.shape[1])]
        else:
            # Single layer (backward compatibility)
            base_tokens = tokens
            all_layer_tokens = [tokens]

        # Determine if we should update weights this step (gradient accumulation)
        is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps != 0

        # Train base transformer (skip if residual-only mode)
        base_loss_val = 0.0
        acc = 0.0
        if not train_residual_only:
            # Use AMP if enabled
            with autocast(device, enabled=use_amp):
                base_loss, pred_ids, acc = base_model(
                    base_tokens,
                    texts,
                    lengths,
                    full_mask_prob=full_mask_prob,
                    label_smoothing=label_smoothing,
                )
                # Scale loss for gradient accumulation
                base_loss = base_loss / gradient_accumulation_steps

            if use_amp and scaler is not None:
                scaler.scale(base_loss).backward()
                if not is_accumulation_step:
                    scaler.unscale_(base_optimizer)
                    torch.nn.utils.clip_grad_norm_(base_model.parameters_wo_clip(), 1.0)
                    scaler.step(base_optimizer)
                    scaler.update()
                    base_optimizer.zero_grad()
            else:
                base_loss.backward()
                if not is_accumulation_step:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters_wo_clip(), 1.0)
                    base_optimizer.step()
                    base_optimizer.zero_grad()

            base_loss_val = base_loss.item() * gradient_accumulation_steps
            accumulated_loss += base_loss_val
            total_loss += base_loss_val
            total_acc += acc
            num_batches += 1

        # Train residual transformer (if enabled and model available)
        res_loss_val = 0.0
        res_acc_val = 0.0
        if train_residual and residual_model is not None and vq_model is not None:
            assert residual_optimizer, "Residual optimizer is not initialized"

            # Randomly select a residual layer to train (1 to num_quantizers-1)
            num_quantizers = len(all_layer_tokens)
            layer_idx = random.randint(1, num_quantizers - 1)

            # Previous layers (0 to layer_idx-1)
            prev_layer_tokens = all_layer_tokens[:layer_idx]
            # Target layer
            target_tokens = all_layer_tokens[layer_idx]

            # Forward pass with AMP
            with autocast(device, enabled=use_amp):
                res_loss, res_pred_ids, res_acc = residual_model(
                    prev_layer_tokens=prev_layer_tokens,
                    target_tokens=target_tokens,
                    layer_idx=layer_idx,
                    texts=texts,
                    m_lens=lengths,
                    vq_model=vq_model,
                )
                # Scale loss for gradient accumulation
                res_loss = res_loss / gradient_accumulation_steps

            if use_amp and scaler is not None:
                scaler.scale(res_loss).backward()
                if not is_accumulation_step:
                    scaler.unscale_(residual_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        residual_model.parameters_wo_clip(), 1.0
                    )
                    scaler.step(residual_optimizer)
                    scaler.update()
                    residual_optimizer.zero_grad()
            else:
                res_loss.backward()
                if not is_accumulation_step:
                    torch.nn.utils.clip_grad_norm_(
                        residual_model.parameters_wo_clip(), 1.0
                    )
                    residual_optimizer.step()
                    residual_optimizer.zero_grad()

            res_loss_val = res_loss.item() * gradient_accumulation_steps
            accumulated_res_loss += res_loss_val
            total_res_loss += res_loss_val
            total_res_acc += res_acc
            num_res_batches += 1
            res_acc_val = res_acc

        # Update progress bar
        postfix = {}
        if not train_residual_only:
            postfix["loss"] = f"{base_loss_val:.4f}"
            postfix["acc"] = f"{acc:.4f}"
        if res_loss_val > 0:
            postfix["res_loss"] = f"{res_loss_val:.4f}"
            postfix["res_acc"] = f"{res_acc_val:.4f}"
        if use_amp:
            postfix["amp"] = "on"
        pbar.set_postfix(postfix)

    # Step schedulers
    if base_scheduler is not None:
        base_scheduler.step()
    if residual_scheduler is not None:
        residual_scheduler.step()

    # Compute averages
    metrics = {}

    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        metrics["loss"] = avg_loss
        metrics["accuracy"] = avg_acc
    else:
        avg_loss = 0.0
        avg_acc = 0.0

    if num_res_batches > 0:
        avg_res_loss = total_res_loss / num_res_batches
        avg_res_acc = total_res_acc / num_res_batches
        metrics["residual_loss"] = avg_res_loss
        metrics["residual_accuracy"] = avg_res_acc

    # Log to TensorBoard
    if writer:
        if num_batches > 0:
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/accuracy", avg_acc, epoch)
            writer.add_scalar("train/lr", base_optimizer.param_groups[0]["lr"], epoch)
        if num_res_batches > 0:
            writer.add_scalar("train/residual_loss", avg_res_loss, epoch)
            writer.add_scalar("train/residual_accuracy", avg_res_acc, epoch)

    return metrics


@torch.no_grad()
def validate(
    model: MaskTransformer,
    dataloader: DataLoader,
    device: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    vq_model=None,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    generate_motions: bool = False,
    num_generation_samples: int = 10,
) -> dict:
    """
    Validate the model.

    Args:
        model: MaskTransformer model
        dataloader: Validation dataloader
        device: Device to run on
        epoch: Current epoch
        writer: TensorBoard writer
        vq_model: RVQ-VAE model (for motion generation if generate_motions=True)
        mean: Normalization mean (for motion generation)
        std: Normalization std (for motion generation)
        generate_motions: Whether to generate actual motions for evaluation
        num_generation_samples: Number of samples to generate for motion metrics
    """
    model.eval()

    total_loss = 0
    total_acc = 0
    num_batches = 0

    # Motion-level metrics (if generating)
    motion_mse_list = []
    motion_mae_list = []

    for batch_idx, (texts, tokens, lengths) in enumerate(
        tqdm(dataloader, desc="Validating")
    ):
        tokens = tokens.to(device)
        lengths = lengths.to(device)

        # Handle multi-layer tokens: (B, num_quantizers, seq_len) or (B, seq_len)
        if len(tokens.shape) == 3:
            # Multi-layer: extract base layer for validation
            base_tokens = tokens[:, 0, :]  # (B, seq_len)
        else:
            # Single layer (backward compatibility)
            base_tokens = tokens

        # Token-level metrics (on base layer)
        loss, pred_ids, acc = model(base_tokens, texts, lengths)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

        # Motion-level metrics (on subset of batches)
        if (
            generate_motions
            and vq_model is not None
            and mean is not None
            and std is not None
        ):
            if batch_idx < num_generation_samples:
                try:
                    # Generate motions
                    gen_tokens = model.generate(
                        texts=texts,
                        m_lens=lengths,
                        timesteps=10,
                        cond_scale=4.0,
                        temperature=1.0,
                    )

                    # Decode generated tokens to motion
                    gen_tokens = torch.clamp(gen_tokens, min=0)
                    gen_tokens_list = [gen_tokens]  # Only base layer for now

                    # Decode using VQ model
                    vq_model.eval()
                    gen_quantized = vq_model.rvq.quantize_from_tokens(gen_tokens_list)
                    gen_motions = vq_model.decoder(gen_quantized)  # (B, D, N)
                    gen_motions = (
                        gen_motions.permute(0, 2, 1).cpu().numpy()
                    )  # (B, N, D)

                    # Decode ground truth tokens
                    # Handle multi-layer tokens: (B, num_quantizers, seq_len) or (B, seq_len)
                    if len(tokens.shape) == 3:
                        # Multi-layer: use all layers
                        gt_tokens_list = [
                            tokens[:, i, :] for i in range(tokens.shape[1])
                        ]
                    else:
                        # Single layer (backward compatibility)
                        gt_tokens_list = [tokens]

                    gt_quantized = vq_model.rvq.quantize_from_tokens(gt_tokens_list)
                    gt_motions = vq_model.decoder(gt_quantized)  # (B, D, N)
                    gt_motions = gt_motions.permute(0, 2, 1).cpu().numpy()  # (B, N, D)

                    # Denormalize
                    gen_motions = gen_motions * std + mean
                    gt_motions = gt_motions * std + mean

                    # Compute metrics on valid frames only
                    for i in range(len(texts)):
                        valid_len = lengths[i].item() * vq_model.downsampling_ratio
                        gen_motion = gen_motions[i, :valid_len]
                        gt_motion = gt_motions[i, :valid_len]

                        # MSE and MAE
                        mse = np.mean((gen_motion - gt_motion) ** 2)
                        mae = np.mean(np.abs(gen_motion - gt_motion))

                        motion_mse_list.append(mse)
                        motion_mae_list.append(mae)

                except Exception as e:
                    print(
                        f"Warning: Motion generation failed for batch {batch_idx}: {e}"
                    )
                    continue

    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
    else:
        avg_loss = 0.0
        avg_acc = 0.0

    metrics = {"loss": avg_loss, "accuracy": avg_acc}

    # Add motion-level metrics if available
    if motion_mse_list:
        metrics["motion_mse"] = np.mean(motion_mse_list)
        metrics["motion_mae"] = np.mean(motion_mae_list)
        metrics["motion_rmse"] = np.sqrt(metrics["motion_mse"])

    if writer:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/accuracy", avg_acc, epoch)
        if "motion_mse" in metrics:
            writer.add_scalar("val/motion_mse", metrics["motion_mse"], epoch)
            writer.add_scalar("val/motion_mae", metrics["motion_mae"], epoch)
            writer.add_scalar("val/motion_rmse", metrics["motion_rmse"], epoch)

    return metrics
