"""
Trainer for Structured LEGO Decoder V2

Tests GENERALIZATION: train on N = 2, 4, 6, 8; test on N = 3, 5, 7

Key changes from V1:
- Continuous N input (not one-hot)
- Latent z for variation
- Generalization test to unseen N values
"""

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange
import logging
import numpy as np

from models.structured_lego import StructuredLegoModelV2, sdf_box
from train.losses_structured import loss_param_supervision


class StructuredLegoTrainerV2:
    """
    Trainer for the structured LEGO decoder V2.
    
    Key difference: uses continuous N input, tests generalization.
    
    Training:
    - Sample N from train_n_values (e.g., [2, 4, 6, 8])
    - Predict half_L = N * stud_spacing / 2 (analytically derived)
    - Learn small offsets for half_H, half_D, stud params via z
    
    Testing:
    - Evaluate on held-out N values (e.g., [3, 5, 7])
    - Check if half_L generalizes correctly
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('structured_trainer_v2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Train/test split for N values
        self.train_n_values = config.get('train_n_values', [2, 4, 6, 8])
        self.test_n_values = config.get('test_n_values', [3, 5, 7])
        
        # Base dimensions
        self.stud_spacing = config.get('stud_spacing', 0.25)
        self.base_half_H = config.get('base_half_H', 0.15)
        self.base_half_D = config.get('base_half_D', 0.125)
        self.base_stud_radius = config.get('base_stud_radius', 0.10)
        self.base_stud_height = config.get('base_stud_height', 0.08)
        
        # Build model
        model_cfg = config.get('model', {})
        self.z_dim = model_cfg.get('z_dim', 4)
        
        self.model = StructuredLegoModelV2(
            z_dim=self.z_dim,
            hidden_dims=model_cfg.get('hidden_dims', [64, 64]),
            max_studs=model_cfg.get('max_studs', 12),
            smooth_k=model_cfg.get('smooth_k', 0.02),
            use_studs=model_cfg.get('use_studs', False),  # box only for now
            base_half_H=self.base_half_H,
            base_half_D=self.base_half_D,
            stud_spacing=self.stud_spacing,
            base_stud_radius=self.base_stud_radius,
            base_stud_height=self.base_stud_height
        ).to(self.device)
        
        # Training config
        train_cfg = config.get('train', {})
        self.epochs = train_cfg.get('epochs', 500)
        self.lr = train_cfg.get('lr', 1e-3)
        self.batch_size = config.get('batch_size', 16)
        
        # Loss weights
        self.lambda_box = config.get('lambda_box', 10.0)
        self.lambda_sdf = config.get('lambda_sdf', 1.0)
        
        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-5
        )
        
        # Checkpoint
        self.ckpt_path = config.get('checkpoint_path', 'checkpoints/structured_lego_v2')
        os.makedirs(self.ckpt_path, exist_ok=True)
        
        self.logger.info(f'StructuredLegoTrainerV2 initialized')
        self.logger.info(f'  Train N: {self.train_n_values}')
        self.logger.info(f'  Test N (held out): {self.test_n_values}')
        self.logger.info(f'  Model params: {sum(p.numel() for p in self.model.parameters())}')
    
    def sample_training_batch(self):
        """Sample batch of (n_studs, z) for training."""
        # Random N from training set
        n_indices = torch.randint(0, len(self.train_n_values), (self.batch_size,))
        n_studs = torch.tensor([self.train_n_values[i] for i in n_indices], 
                               device=self.device, dtype=torch.float32).unsqueeze(-1)  # (B, 1)
        
        # Random latent z
        z = torch.randn(self.batch_size, self.z_dim, device=self.device) * 0.5  # (B, z_dim)
        
        return n_studs, z
    
    def get_target_half_L(self, n_studs):
        """Compute target half_L from N (analytic formula)."""
        return n_studs * self.stud_spacing / 2  # (B, 1)
    
    def sample_points(self, n_studs, n_points=1024):
        """Sample points inside/outside the expected box."""
        B = n_studs.shape[0]
        
        # Max extent based on max N in batch
        max_half_L = (n_studs.max() * self.stud_spacing / 2 + 0.2).item()
        max_extent = max(max_half_L, self.base_half_H + 0.2, self.base_half_D + 0.2)
        
        # Random points in [-max_extent, max_extent]^3
        points = (torch.rand(B, n_points, 3, device=self.device) * 2 - 1) * max_extent
        
        return points
    
    def compute_target_sdf(self, points, n_studs):
        """Compute ground-truth SDF for box with given N."""
        B = points.shape[0]
        
        # Target half-extents
        half_L = self.get_target_half_L(n_studs)  # (B, 1)
        half_H = torch.full((B, 1), self.base_half_H, device=self.device)
        half_D = torch.full((B, 1), self.base_half_D, device=self.device)
        half_extents = torch.cat([half_L, half_H, half_D], dim=-1)  # (B, 3)
        
        return sdf_box(points, half_extents)  # (B, N_pts)
    
    def train_step(self):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Sample batch
        n_studs, z = self.sample_training_batch()
        
        # Get predicted parameters
        params = self.model.get_params(n_studs, z)
        
        # Target half_L (analytical)
        target_half_L = self.get_target_half_L(n_studs)
        
        # Loss on half_L (the key thing to get right)
        loss_L = F.mse_loss(params['half_extents'][:, 0:1], target_half_L)
        
        # Loss on half_H, half_D (should stay near base values)
        loss_H = F.mse_loss(params['half_extents'][:, 1:2], 
                           torch.full_like(target_half_L, self.base_half_H))
        loss_D = F.mse_loss(params['half_extents'][:, 2:3], 
                           torch.full_like(target_half_L, self.base_half_D))
        
        loss_box = loss_L + 0.5 * (loss_H + loss_D)
        
        # SDF supervision
        points = self.sample_points(n_studs, n_points=512)
        pred_sdf = self.model(points, n_studs, z)
        target_sdf = self.compute_target_sdf(points, n_studs)
        loss_sdf = F.mse_loss(pred_sdf, target_sdf)
        
        # Total loss
        loss_total = self.lambda_box * loss_box + self.lambda_sdf * loss_sdf
        
        loss_total.backward()
        self.optimizer.step()
        
        return {
            'loss_total': loss_total.item(),
            'loss_box': loss_box.item(),
            'loss_L': loss_L.item(),
            'loss_sdf': loss_sdf.item()
        }
    
    def evaluate(self, n_values, name=''):
        """Evaluate on a set of N values."""
        self.model.eval()
        
        results = {}
        with torch.no_grad():
            for n in n_values:
                n_studs = torch.tensor([[float(n)]], device=self.device)
                z = torch.zeros(1, self.z_dim, device=self.device)
                
                params = self.model.get_params(n_studs, z)
                
                pred_half_L = params['half_extents'][0, 0].item()
                pred_half_H = params['half_extents'][0, 1].item()
                pred_half_D = params['half_extents'][0, 2].item()
                
                target_half_L = n * self.stud_spacing / 2
                
                error_L = abs(pred_half_L - target_half_L)
                error_rel = error_L / target_half_L * 100
                
                results[n] = {
                    'pred_half_L': pred_half_L,
                    'target_half_L': target_half_L,
                    'error_L': error_L,
                    'error_rel': error_rel,
                    'pred_half_H': pred_half_H,
                    'pred_half_D': pred_half_D
                }
        
        return results
    
    def train(self):
        """Full training loop with generalization testing."""
        self.logger.info(f'Starting training for {self.epochs} epochs')
        
        pbar = trange(self.epochs, desc='Training')
        for epoch in pbar:
            losses = self.train_step()
            self.scheduler.step()
            
            pbar.set_postfix({
                'loss': f"{losses['loss_total']:.4f}",
                'L_err': f"{losses['loss_L']:.5f}"
            })
            
            # Periodic evaluation
            if (epoch + 1) % 50 == 0:
                self.logger.info(f'\n=== Epoch {epoch+1} ===')
                
                # Train set
                train_results = self.evaluate(self.train_n_values, 'train')
                self.logger.info('Training N values:')
                for n, r in train_results.items():
                    self.logger.info(f"  N={n}: pred_L={r['pred_half_L']:.4f}, "
                                   f"target_L={r['target_half_L']:.4f}, "
                                   f"error={r['error_rel']:.2f}%")
                
                # Test set (generalization)
                test_results = self.evaluate(self.test_n_values, 'test')
                self.logger.info('Held-out N values (GENERALIZATION TEST):')
                for n, r in test_results.items():
                    status = "✓" if r['error_rel'] < 5 else "✗"
                    self.logger.info(f"  N={n}: pred_L={r['pred_half_L']:.4f}, "
                                   f"target_L={r['target_half_L']:.4f}, "
                                   f"error={r['error_rel']:.2f}% {status}")
        
        # Final evaluation
        self.logger.info('\n' + '='*50)
        self.logger.info('FINAL GENERALIZATION RESULTS')
        self.logger.info('='*50)
        
        test_results = self.evaluate(self.test_n_values)
        avg_error = np.mean([r['error_rel'] for r in test_results.values()])
        
        for n, r in test_results.items():
            status = "PASS" if r['error_rel'] < 5 else "FAIL"
            self.logger.info(f"N={n}: error={r['error_rel']:.2f}% [{status}]")
        
        self.logger.info(f'\nAverage generalization error: {avg_error:.2f}%')
        
        if avg_error < 5:
            self.logger.info('✓ Model GENERALIZES to unseen N values!')
        else:
            self.logger.info('✗ Model does NOT generalize well.')
        
        # Save
        self.save_checkpoint(self.epochs)
    
    def save_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_path, f'model_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        self.logger.info(f'Saved: {path}')


def test_interpolation_v2(model, stud_spacing, n_start=2, n_end=8, n_steps=7, device='cuda'):
    """
    Test continuous N interpolation.
    
    For structured decoder V2, half_L should scale linearly with N.
    """
    model.eval()
    z_dim = model.z_dim
    
    print("\n" + "="*50)
    print("CONTINUOUS N INTERPOLATION TEST")
    print("="*50)
    
    with torch.no_grad():
        n_values = torch.linspace(n_start, n_end, n_steps)
        
        for n in n_values:
            n_studs = torch.tensor([[n.item()]], device=device)
            z = torch.zeros(1, z_dim, device=device)
            
            params = model.get_params(n_studs, z)
            pred_half_L = params['half_extents'][0, 0].item()
            expected_half_L = n.item() * stud_spacing / 2
            error = abs(pred_half_L - expected_half_L) / expected_half_L * 100
            
            bar_len = int(pred_half_L * 40)
            bar = '█' * bar_len
            
            print(f"N={n.item():.1f}: half_L={pred_half_L:.3f} (exp={expected_half_L:.3f}) "
                  f"err={error:.1f}% |{bar}")


if __name__ == '__main__':
    import yaml
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Default config for V2
    config = {
        'train_n_values': [2, 4, 6, 8],  # train on these
        'test_n_values': [3, 5, 7],       # test generalization on these
        'stud_spacing': 0.25,
        'base_half_H': 0.15,
        'base_half_D': 0.125,
        'batch_size': 32,
        'model': {
            'z_dim': 4,
            'hidden_dims': [64, 64],
            'max_studs': 12,
            'use_studs': False  # box only
        },
        'train': {
            'epochs': 300,
            'lr': 1e-3
        },
        'lambda_box': 10.0,
        'lambda_sdf': 1.0,
        'checkpoint_path': 'checkpoints/structured_lego_v2'
    }
    
    # Override with config file if provided
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            config.update(yaml.safe_load(f))
    
    # Train
    trainer = StructuredLegoTrainerV2(config)
    trainer.train()
    
    # Test interpolation
    test_interpolation_v2(
        trainer.model, 
        trainer.stud_spacing,
        n_start=2, n_end=8, n_steps=13,
        device=trainer.device
    )
