"""
Training utilities for molecular diffusion model.
"""
import os
import copy
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
from model.diffusion import MolecularDiffusion


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights (for evaluation/sampling)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original weights after EMA evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            scale = self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            import math
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']


class Trainer:
    """
    Trainer class for molecular diffusion model.

    Handles training loop, validation, checkpointing, EMA, and logging.
    """

    def __init__(
        self,
        model: MolecularDiffusion,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Move model to device
        self.device = config.device
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler with warmup
        total_steps = config.training.epochs * len(train_loader)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.training.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.01,
        )

        # EMA
        self.ema = EMA(model, decay=config.training.ema_decay)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'node_acc': [],
            'edge_acc': [],
            'valency_loss': [],
            'property_loss': [],
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_node_loss = 0.0
        total_edge_loss = 0.0
        total_valency_loss = 0.0
        total_property_loss = 0.0
        total_node_acc = 0.0
        total_edge_acc = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            # Move batch to device
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            metrics = self.model.training_step(batch)

            # Backward pass
            loss = metrics['loss']
            loss.backward()

            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )

            self.optimizer.step()
            self.scheduler.step()
            self.ema.update(self.model)

            # Update metrics
            total_loss += loss.item()
            total_node_loss += metrics['node_loss'].item()
            total_edge_loss += metrics['edge_loss'].item()
            total_valency_loss += metrics['valency_loss'].item()
            total_property_loss += metrics['property_loss'].item()
            total_node_acc += metrics['node_acc'].item()
            total_edge_acc += metrics['edge_acc'].item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'node_acc': f"{metrics['node_acc'].item():.3f}",
                'val_l': f"{metrics['valency_loss'].item():.3f}",
                'prop_l': f"{metrics['property_loss'].item():.3f}",
            })

            # Periodic detailed loss breakdown
            if num_batches % 10 == 0:
                diff_l = metrics.get('diffusion_loss', metrics['node_loss'] + metrics['edge_loss'])
                diff_val = diff_l.item() if hasattr(diff_l, 'item') else float(diff_l)
                val_l = metrics['valency_loss'].item()
                prop_l = metrics['property_loss'].item()
                pbar.write(
                    f"  [Step {self.global_step}] "
                    f"diff={diff_val:.4f}  val={val_l:.4f}  prop={prop_l:.4f}  "
                    f"ratio(val/diff)={val_l / max(diff_val, 1e-8):.3f}  "
                    f"ratio(prop/diff)={prop_l / max(diff_val, 1e-8):.3f}"
                )

        # Average metrics
        return {
            'loss': total_loss / num_batches,
            'node_loss': total_node_loss / num_batches,
            'edge_loss': total_edge_loss / num_batches,
            'valency_loss': total_valency_loss / num_batches,
            'property_loss': total_property_loss / num_batches,
            'node_acc': total_node_acc / num_batches,
            'edge_acc': total_edge_acc / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model using EMA weights."""
        if self.val_loader is None:
            return {}

        # Use EMA weights for validation
        self.ema.apply_shadow(self.model)
        self.model.eval()

        total_loss = 0.0
        total_node_acc = 0.0
        total_edge_acc = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            metrics = self.model.training_step(batch)

            total_loss += metrics['loss'].item()
            total_node_acc += metrics['node_acc'].item()
            total_edge_acc += metrics['edge_acc'].item()
            num_batches += 1

        # Restore original weights
        self.ema.restore(self.model)

        return {
            'val_loss': total_loss / num_batches,
            'val_node_acc': total_node_acc / num_batches,
            'val_edge_acc': total_edge_acc / num_batches,
        }

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, list]:
        """Full training loop."""
        num_epochs = num_epochs or self.config.training.epochs

        print(f"Starting training for {num_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")
        print(f"EMA decay: {self.config.training.ema_decay}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(val_loss: {train_metrics['valency_loss']:.3f}, "
                  f"prop_loss: {train_metrics['property_loss']:.3f})")
            print(f"  Node Acc: {train_metrics['node_acc']:.3f}")
            print(f"  Edge Acc: {train_metrics['edge_acc']:.3f}")

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['node_acc'].append(train_metrics['node_acc'])
            self.history['edge_acc'].append(train_metrics['edge_acc'])
            self.history['valency_loss'].append(train_metrics['valency_loss'])
            self.history['property_loss'].append(train_metrics['property_loss'])

            # Validate
            if self.val_loader and (epoch + 1) % self.config.training.val_frequency == 0:
                val_metrics = self.validate()
                print(f"  Val Loss (EMA): {val_metrics['val_loss']:.4f}")
                print(f"  Val Node Acc: {val_metrics['val_node_acc']:.3f}")

                self.history['val_loss'].append(val_metrics['val_loss'])

                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pt')
                    print("  Saved best model!")

            # Periodic checkpoint
            if (epoch + 1) % self.config.training.save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        # Save final model
        self.save_checkpoint('final_model.pt')
        print("\nTraining complete!")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint including EMA state."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint including EMA state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
