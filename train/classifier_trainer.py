# train/classifier_trainer.py
"""
Training pipeline for molecular glue classifier.
Similar to diffusion trainer but simpler: BCEWithLogitsLoss, standard metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import os

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class ClassifierTrainer:
    """Training pipeline for molecular glue classifier."""
    
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            num_workers=config.num_workers
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=config.num_workers
        )
        
        # Optimizer
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
        
        # Loss function (Binary Cross-Entropy)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Tracking
        self.best_val_acc = 0.0
        self.epoch = 0
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Logging
        if config.use_wandb and HAS_WANDB:
            wandb.init(
                project=getattr(config, 'project_name', 'molecular-glue-classifier'),
                config=vars(config) if hasattr(config, '__dict__') else {}
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            logits = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
            
            # Compute loss
            loss = self.criterion(logits, batch.y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
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
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = batch.to(self.device)
                
                # Forward pass
                logits = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
                
                # Compute loss
                loss = self.criterion(logits, batch.y)
                total_loss += loss.item()
                
                # Track predictions
                probs = torch.sigmoid(logits).cpu().numpy()
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
            auc = 0.0  # Fails if only one class in batch
        
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
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['accuracy'])
            
            # Print metrics
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"          Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            print(f"          F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # Log to wandb
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
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('best_classifier.pt')
                print(f"  ✓ New best model! (val_acc={val_metrics['accuracy']:.4f})")
        
        print(f"\n✓ Training complete! Best val accuracy: {self.best_val_acc:.4f}")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': {
                'hidden_dim': self.model.hidden_dim,
                'num_layers': len(self.model.layers),
                'num_heads': self.model.layers[0].local_attn.num_heads,
                'dropout': self.config.dropout,
            },
        }
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved to {path}")