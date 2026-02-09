"""
Training utilities for molecular diffusion model.
"""
import os
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


class Trainer:
    """
    Trainer class for molecular diffusion model.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: MolecularDiffusion,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Args:
            model: MolecularDiffusion model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
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
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.learning_rate * 0.01,
        )
        
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
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_node_loss = 0.0
        total_edge_loss = 0.0
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
            
            # Update metrics
            total_loss += loss.item()
            total_node_loss += metrics['node_loss'].item()
            total_edge_loss += metrics['edge_loss'].item()
            total_node_acc += metrics['node_acc'].item()
            total_edge_acc += metrics['edge_acc'].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'node_acc': f"{metrics['node_acc'].item():.3f}",
            })
        
        # Average metrics
        return {
            'loss': total_loss / num_batches,
            'node_loss': total_node_loss / num_batches,
            'edge_loss': total_edge_loss / num_batches,
            'node_acc': total_node_acc / num_batches,
            'edge_acc': total_edge_acc / num_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
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
        
        return {
            'val_loss': total_loss / num_batches,
            'val_node_acc': total_node_acc / num_batches,
            'val_edge_acc': total_edge_acc / num_batches,
        }
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs (uses config if not specified)
            
        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.training.epochs
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Node Acc: {train_metrics['node_acc']:.3f}")
            print(f"  Edge Acc: {train_metrics['edge_acc']:.3f}")
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['node_acc'].append(train_metrics['node_acc'])
            self.history['edge_acc'].append(train_metrics['edge_acc'])
            
            # Validate
            if self.val_loader and (epoch + 1) % self.config.training.val_frequency == 0:
                val_metrics = self.validate()
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Val Node Acc: {val_metrics['val_node_acc']:.3f}")
                
                self.history['val_loss'].append(val_metrics['val_loss'])
                
                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pt')
                    print("  Saved best model!")
            
            # Update learning rate
            self.scheduler.step()
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.training.save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        print("\nTraining complete!")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
