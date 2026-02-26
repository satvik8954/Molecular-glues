# train/classifier_trainer.py
"""
Training pipeline for molecular glue classifier.
Supports: single GPU, DataParallel, and DistributedDataParallel (DDP).

Usage:
  Single GPU:  python train_classifier.py --epochs 50
  Multi-GPU:   torchrun --nproc_per_node=NUM_GPUS train_classifier.py --epochs 50
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import os

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def _get_raw_model(model):
    """Unwrap model from DataParallel/DDP wrappers."""
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


class ClassifierTrainer:
    """
    Training pipeline for molecular glue classifier.
    
    Automatically selects the best parallelism strategy:
      - CPU: single process
      - 1 GPU: single GPU with AMP
      - N GPUs + torchrun: DistributedDataParallel (fastest)
      - N GPUs + python: DataParallel (fallback)
    """
    
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.config = config
        
        # ---- Detect distributed environment ----
        self.is_ddp = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_ddp else 0
        self.world_size = dist.get_world_size() if self.is_ddp else 1
        self.is_main = (self.rank == 0)
        
        # ---- Device setup ----
        if self.is_ddp:
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enable cuDNN auto-tuner
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        self.model.to(self.device)
        
        # ---- Parallelism strategy ----
        self.num_gpus = torch.cuda.device_count() if self.device.type == 'cuda' else 0
        
        if self.is_ddp:
            # DistributedDataParallel (one process per GPU — fastest)
            self.model = DDP(self.model, device_ids=[self.local_rank], 
                           output_device=self.local_rank,
                           find_unused_parameters=False)
            if self.is_main:
                print(f"  ⚡ Using DistributedDataParallel with {self.world_size} GPUs")
        elif self.num_gpus > 1:
            # DataParallel fallback (single process, all GPUs)
            self.model = nn.DataParallel(self.model)
            config.batch_size = config.batch_size * self.num_gpus
            if self.is_main:
                print(f"  Using DataParallel with {self.num_gpus} GPUs")
        
        # ---- Mixed precision ----
        self.use_amp = self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        # ---- Data loaders ----
        use_pin = self.device.type == 'cuda'
        use_persistent = config.num_workers > 0
        
        # DDP requires DistributedSampler (splits data across GPUs)
        self.train_sampler = (
            DistributedSampler(train_dataset, num_replicas=self.world_size, 
                              rank=self.rank, shuffle=True)
            if self.is_ddp else None
        )
        self.val_sampler = (
            DistributedSampler(val_dataset, num_replicas=self.world_size,
                              rank=self.rank, shuffle=False)
            if self.is_ddp else None
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(self.train_sampler is None),  # sampler handles shuffle for DDP
            sampler=self.train_sampler,
            collate_fn=train_dataset.collate_fn,
            num_workers=config.num_workers,
            pin_memory=use_pin,
            persistent_workers=use_persistent,
            prefetch_factor=4 if config.num_workers > 0 else None,
            drop_last=self.is_ddp  # avoid uneven batch sizes in DDP
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            collate_fn=val_dataset.collate_fn,
            num_workers=config.num_workers,
            pin_memory=use_pin,
            persistent_workers=use_persistent,
            prefetch_factor=4 if config.num_workers > 0 else None
        )
        
        # ---- Optimizer ----
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Tracking
        self.best_val_acc = 0.0
        self.epoch = 0
        
        # Create checkpoint directory (only on main process)
        if self.is_main:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Logging (only on main process)
        if self.is_main and config.use_wandb and HAS_WANDB:
            wandb.init(
                project=getattr(config, 'project_name', 'molecular-glue-classifier'),
                config=vars(config) if hasattr(config, '__dict__') else {}
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def _log(self, msg):
        """Print only on main process."""
        if self.is_main:
            print(msg)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        # Set epoch for DistributedSampler (ensures proper shuffling)
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}', 
                    disable=not self.is_main)
        
        for batch in pbar:
            batch = batch.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
                loss = self.criterion(logits, batch.y)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Track metrics
            total_loss += loss.item()
            
            preds = torch.sigmoid(logits).detach().cpu() > 0.5
            all_preds.extend(preds.numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation', 
                            disable=not self.is_main):
                batch = batch.to(self.device, non_blocking=True)
                
                # Forward pass (AMP for validation too)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
                    loss = self.criterion(logits, batch.y)
                
                total_loss += loss.item()
                
                # Track predictions
                probs = torch.sigmoid(logits).float().cpu().numpy()
                preds = probs > 0.5
                
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(batch.y.cpu().numpy())
        
        # Compute metrics (with safety for edge cases)
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        try:
            precision = precision_score(all_labels, all_preds, zero_division=0)
        except Exception:
            precision = 0.0
        
        try:
            recall = recall_score(all_labels, all_preds, zero_division=0)
        except Exception:
            recall = 0.0
        
        try:
            f1 = f1_score(all_labels, all_preds, zero_division=0)
        except Exception:
            f1 = 0.0
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train(self, num_epochs):
        """Main training loop."""
        self._log(f"Starting training for {num_epochs} epochs...")
        self._log(f"Device: {self.device}")
        if self.is_ddp:
            self._log(f"Mode: DistributedDataParallel ({self.world_size} processes)")
        elif self.num_gpus > 1:
            self._log(f"Mode: DataParallel ({self.num_gpus} GPUs)")
        else:
            self._log(f"Mode: Single {'GPU' if self.device.type == 'cuda' else 'CPU'}")
        
        raw_model = _get_raw_model(self.model)
        total_params = sum(p.numel() for p in raw_model.parameters())
        self._log(f"Model parameters: {total_params:,}")
        self._log(f"Batch size per GPU: {self.config.batch_size}")
        self._log(f"Effective batch size: {self.config.batch_size * self.world_size}")
        self._log(f"Data workers: {self.config.num_workers}")
        self._log(f"Mixed precision: {self.use_amp}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['accuracy'])
            
            # Print metrics (main process only)
            self._log(f"\nEpoch {epoch}:")
            self._log(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            self._log(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            self._log(f"          Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            self._log(f"          F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # Log to wandb (main process only)
            if self.use_wandb:
                wandb.log({
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'val/f1': val_metrics['f1'],
                    'val/auc': val_metrics['auc'],
                    'epoch': epoch
                })
            
            # Save best model (main process only)
            if self.is_main and val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('best_classifier.pt')
                self._log(f"  ✓ New best model! (val_acc={val_metrics['accuracy']:.4f})")
            
            # Synchronize across processes
            if self.is_ddp:
                dist.barrier()
        
        self._log(f"\n✓ Training complete! Best val accuracy: {self.best_val_acc:.4f}")
        
        # Cleanup DDP
        if self.is_ddp:
            dist.destroy_process_group()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint (unwraps DataParallel/DDP)."""
        raw_model = _get_raw_model(self.model)
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': {
                'hidden_dim': raw_model.hidden_dim,
                'num_layers': len(raw_model.layers),
                'num_heads': raw_model.layers[0].local_attn.num_heads,
                'dropout': self.config.dropout,
            },
        }
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        self._log(f"  Checkpoint saved to {path}")