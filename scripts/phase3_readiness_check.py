"""
Phase 3 → Phase 4 Readiness Check

This script runs a short training session and evaluates the rubric:
1. Training stability (no NaNs, no exploding losses)
2. Connectivity actually helps (component count or |f(saddle)| drops)
3. Diversity doesn't destroy validity

Run: python scripts/phase3_readiness_check.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import time
from collections import defaultdict

# Project imports
from util.checkpointing import load_yaml_and_drop_keys
from util.misc import get_model
from models.net_w_partials import NetWithPartials
from train.losses_connectivity import (
    find_critical_points_newton,
    classify_critical_points,
    CachedConnectivityLoss,
)
from train.losses_diversity import diversity_loss_chamfer
from models.point_wrapper import PointWrapper
from skimage import measure

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================
# Configuration
# ============================================================
N_STEPS = 2000  # Number of training steps
LOG_EVERY = 100  # Log metrics every N steps
N_FIXED_LATENTS = 10  # Fixed latents for tracking
LEARNING_RATE = 1e-4

# Loss weights (start conservative)
LAMBDA_SDF = 1.0
LAMBDA_DIVERSITY = 0.01  # Start small
LAMBDA_CONNECTIVITY = 0.001  # Start very small

# Connectivity settings
CONNECTIVITY_UPDATE_EVERY = 50  # Update saddle cache every N steps
CONNECTIVITY_N_SAMPLES = 500
CONNECTIVITY_N_ITER = 30

# ============================================================
# Load Model
# ============================================================
print("\n" + "="*60)
print("Loading GINN model...")
print("="*60)

config = load_yaml_and_drop_keys('checkpoints/GINN-config.yml', keys_to_drop=[])
bounds = torch.from_numpy(np.load('GINN/simJEB/data/bounds.npy')).float().to(device)

model_raw = get_model(**config['model'], use_legacy_gabor=True)
model_raw.load_state_dict(torch.load('checkpoints/GINN-model.pt', map_location=device))
model_raw = model_raw.to(device)
model_raw.train()

nz = config['nz']
print(f"Model loaded. nz={nz}, bounds={bounds.cpu().numpy()}")

# ============================================================
# Fixed Latents for Tracking
# ============================================================
print(f"\nCreating {N_FIXED_LATENTS} fixed latents for tracking...")
fixed_latents = torch.linspace(-0.8, 0.8, N_FIXED_LATENTS).unsqueeze(1).repeat(1, nz).to(device)
print(f"Latent range: [{fixed_latents.min():.2f}, {fixed_latents.max():.2f}]")

# ============================================================
# Helper Functions
# ============================================================

def count_components(sdf_grid, level=0.0):
    """Count connected components in SDF grid."""
    interior = (sdf_grid < level)
    labeled, n = ndimage.label(interior)
    return n

def get_sdf_grid(model, z, bounds, resolution=48):
    """Evaluate SDF on a grid for a given z."""
    xs = torch.linspace(bounds[0, 0], bounds[0, 1], resolution, device=device)
    ys = torch.linspace(bounds[1, 0], bounds[1, 1], resolution, device=device)
    zs = torch.linspace(bounds[2, 0], bounds[2, 1], resolution, device=device)
    grid_pts = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1).reshape(-1, 3)
    
    z_batch = z.unsqueeze(0).expand(len(grid_pts), -1)
    with torch.no_grad():
        sdf_vals = model(grid_pts, z_batch).squeeze()
    return sdf_vals.reshape(resolution, resolution, resolution).cpu().numpy()

def compute_metrics_for_latent(model, z, bounds, cached_connectivity):
    """Compute all metrics for a single latent."""
    # Wrapper for fixed z
    class FixedZ:
        def __init__(self, model, z):
            self.model = model
            self.z = z
        def __call__(self, x):
            z_batch = self.z.unsqueeze(0).expand(len(x), -1)
            return self.model(x, z_batch)
    
    f = FixedZ(model, z)
    
    # Component count (grid-based)
    sdf_grid = get_sdf_grid(model, z, bounds, resolution=48)
    n_components = count_components(sdf_grid)
    
    # Check validity: is there any surface?
    has_surface = (sdf_grid.min() < 0) and (sdf_grid.max() > 0)
    
    # Check bounds: what fraction is in-bounds?
    in_bounds_ratio = ((sdf_grid >= -1) & (sdf_grid <= 1)).mean()
    
    # Mean |f(saddle)| if we have cached saddles
    mean_f_saddle = 0.0
    n_saddles = 0
    
    return {
        'n_components': n_components,
        'has_surface': has_surface,
        'in_bounds_ratio': in_bounds_ratio,
        'mean_f_saddle': mean_f_saddle,
        'n_saddles': n_saddles,
    }

def compute_all_fixed_latent_metrics(model, fixed_latents, bounds, cached_connectivity):
    """Compute metrics for all fixed latents."""
    metrics = defaultdict(list)
    for i, z in enumerate(fixed_latents):
        m = compute_metrics_for_latent(model, z, bounds, cached_connectivity)
        for k, v in m.items():
            metrics[k].append(v)
    
    # Aggregate
    return {
        'mean_components': np.mean(metrics['n_components']),
        'max_components': np.max(metrics['n_components']),
        'min_components': np.min(metrics['n_components']),
        'valid_surfaces': sum(metrics['has_surface']),
        'mean_in_bounds': np.mean(metrics['in_bounds_ratio']),
        'component_list': metrics['n_components'],
    }

def sample_surface_points(model, z, bounds, n_points=200, resolution=48):
    """Sample surface points via marching cubes."""
    sdf_grid = get_sdf_grid(model, z, bounds, resolution)
    try:
        verts, faces, _, _ = measure.marching_cubes(sdf_grid, level=0.0)
        verts = verts / (resolution - 1)
        verts = verts * (bounds[:, 1] - bounds[:, 0]).cpu().numpy() + bounds[:, 0].cpu().numpy()
        verts = torch.tensor(verts, dtype=torch.float32, device=device)
        if len(verts) > n_points:
            idx = torch.randperm(len(verts))[:n_points]
            verts = verts[idx]
        return verts
    except:
        return None

# ============================================================
# Training Loop
# ============================================================
print("\n" + "="*60)
print("Starting Phase 3 Readiness Check")
print("="*60)
print(f"Steps: {N_STEPS}, Log every: {LOG_EVERY}")
print(f"λ_diversity={LAMBDA_DIVERSITY}, λ_connectivity={LAMBDA_CONNECTIVITY}")
print("="*60 + "\n")

optimizer = torch.optim.Adam(model_raw.parameters(), lr=LEARNING_RATE)

# Initialize cached connectivity loss
cached_connectivity = CachedConnectivityLoss(
    bounds=bounds,
    n_samples=CONNECTIVITY_N_SAMPLES,
    n_iter=CONNECTIVITY_N_ITER,
    grad_tol=1e-2,
    update_every_n_epochs=CONNECTIVITY_UPDATE_EVERY,
)

# Tracking
history = defaultdict(list)
initial_metrics = None
nan_detected = False
exploding_loss_detected = False

try:
    for step in tqdm(range(N_STEPS), desc="Training"):
        optimizer.zero_grad()
        
        # Sample batch of random z values (generative mode)
        batch_size = 4
        z_batch = torch.rand(batch_size, nz, device=device) * 2 - 1  # [-1, 1]
        
        # Sample random points in bounds
        n_pts = 500
        pts = torch.rand(n_pts, 3, device=device)
        pts = pts * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        
        # Expand for batch
        pts_expanded = pts.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
        z_expanded = z_batch.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, nz)
        
        # Forward pass
        sdf_vals = model_raw(pts_expanded, z_expanded).squeeze()
        
        # Basic SDF loss (just regularization - penalize extreme values)
        loss_sdf = (sdf_vals.abs() - 0.5).relu().mean()
        
        # Diversity loss (every step)
        loss_diversity = torch.tensor(0.0, device=device)
        if LAMBDA_DIVERSITY > 0:
            surface_pts_list = []
            for i in range(batch_size):
                sp = sample_surface_points(model_raw, z_batch[i], bounds, n_points=100)
                if sp is not None and len(sp) > 10:
                    surface_pts_list.append(sp)
            
            if len(surface_pts_list) >= 2:
                pw = PointWrapper.create_from_pts_per_shape_list(surface_pts_list)
                loss_diversity = diversity_loss_chamfer(pw, aggregation='mean')
        
        # Connectivity loss (periodic)
        loss_connectivity = torch.tensor(0.0, device=device)
        if LAMBDA_CONNECTIVITY > 0 and step % 10 == 0:  # Every 10 steps
            # Use first z in batch
            class FixedZ:
                def __init__(self, model, z):
                    self.model = model
                    self.z = z
                def __call__(self, x):
                    z_b = self.z.unsqueeze(0).expand(len(x), -1)
                    return self.model(x, z_b)
            
            f = FixedZ(model_raw, z_batch[0])
            loss_conn, info = cached_connectivity(f, step)
            loss_connectivity = loss_conn
        
        # Total loss
        total_loss = (LAMBDA_SDF * loss_sdf + 
                      LAMBDA_DIVERSITY * loss_diversity + 
                      LAMBDA_CONNECTIVITY * loss_connectivity)
        
        # Check for NaN
        if torch.isnan(total_loss):
            print(f"\n❌ NaN detected at step {step}!")
            nan_detected = True
            break
        
        # Check for exploding loss
        if total_loss.item() > 1e6:
            print(f"\n❌ Exploding loss detected at step {step}: {total_loss.item():.2e}")
            exploding_loss_detected = True
            break
        
        # Backward
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model_raw.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Log losses
        history['step'].append(step)
        history['loss_total'].append(total_loss.item())
        history['loss_sdf'].append(loss_sdf.item())
        history['loss_diversity'].append(loss_diversity.item() if isinstance(loss_diversity, torch.Tensor) else loss_diversity)
        history['loss_connectivity'].append(loss_connectivity.item() if isinstance(loss_connectivity, torch.Tensor) else loss_connectivity)
        
        # Periodic detailed logging
        if step % LOG_EVERY == 0:
            # Compute metrics on fixed latents
            with torch.no_grad():
                metrics = compute_all_fixed_latent_metrics(
                    model_raw, fixed_latents, bounds, cached_connectivity
                )
            
            if initial_metrics is None:
                initial_metrics = metrics.copy()
            
            history['mean_components'].append(metrics['mean_components'])
            history['valid_surfaces'].append(metrics['valid_surfaces'])
            history['mean_in_bounds'].append(metrics['mean_in_bounds'])
            
            print(f"\nStep {step}:")
            print(f"  Losses: total={total_loss.item():.4f}, sdf={loss_sdf.item():.4f}, "
                  f"div={history['loss_diversity'][-1]:.4f}, conn={history['loss_connectivity'][-1]:.6f}")
            print(f"  Components: mean={metrics['mean_components']:.1f}, "
                  f"range=[{metrics['min_components']}, {metrics['max_components']}]")
            print(f"  Valid surfaces: {metrics['valid_surfaces']}/{N_FIXED_LATENTS}")
            print(f"  Component list: {metrics['component_list']}")

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")

# ============================================================
# Evaluation Report
# ============================================================
print("\n" + "="*60)
print("PHASE 3 → PHASE 4 READINESS REPORT")
print("="*60)

# Criterion 1: Training Stability
print("\n📊 CRITERION 1: Training Stability")
print("-" * 40)
if nan_detected:
    print("  ❌ NaN detected during training")
    criterion_1 = False
elif exploding_loss_detected:
    print("  ❌ Exploding loss detected")
    criterion_1 = False
else:
    # Check for reasonable loss values
    final_losses = history['loss_total'][-100:] if len(history['loss_total']) >= 100 else history['loss_total']
    loss_std = np.std(final_losses)
    loss_mean = np.mean(final_losses)
    
    print(f"  ✓ No NaN or exploding losses")
    print(f"  Final 100 steps: mean={loss_mean:.4f}, std={loss_std:.4f}")
    
    # Check if connectivity dominates
    conn_losses = history['loss_connectivity'][-100:] if len(history['loss_connectivity']) >= 100 else history['loss_connectivity']
    conn_ratio = np.mean(conn_losses) * LAMBDA_CONNECTIVITY / (loss_mean + 1e-8)
    print(f"  Connectivity contribution: {conn_ratio*100:.1f}% of total loss")
    
    if conn_ratio > 0.8:
        print("  ⚠️  Warning: Connectivity loss dominates (>80%)")
        criterion_1 = False
    else:
        criterion_1 = True

# Criterion 2: Connectivity Actually Helps
print("\n📊 CRITERION 2: Connectivity Helps in Training")
print("-" * 40)
if len(history['mean_components']) >= 2:
    initial_components = history['mean_components'][0]
    final_components = history['mean_components'][-1]
    component_change = final_components - initial_components
    
    print(f"  Initial mean components: {initial_components:.1f}")
    print(f"  Final mean components: {final_components:.1f}")
    print(f"  Change: {component_change:+.1f}")
    
    if component_change < -0.5:
        print("  ✓ Component count improved!")
        criterion_2 = True
    elif component_change > 1.0:
        print("  ❌ Component count got worse")
        criterion_2 = False
    else:
        print("  ⚠️  No significant change (inconclusive)")
        criterion_2 = None  # Inconclusive
else:
    print("  ⚠️  Not enough data points")
    criterion_2 = None

# Criterion 3: Diversity Doesn't Destroy Validity
print("\n📊 CRITERION 3: Diversity Preserves Validity")
print("-" * 40)
if len(history['valid_surfaces']) >= 2:
    initial_valid = history['valid_surfaces'][0]
    final_valid = history['valid_surfaces'][-1]
    
    print(f"  Initial valid surfaces: {initial_valid}/{N_FIXED_LATENTS}")
    print(f"  Final valid surfaces: {final_valid}/{N_FIXED_LATENTS}")
    
    if final_valid >= initial_valid - 1:
        print("  ✓ Validity preserved")
        criterion_3 = True
    else:
        print(f"  ❌ Lost {initial_valid - final_valid} valid surfaces")
        criterion_3 = False
    
    # Check in-bounds ratio
    if len(history['mean_in_bounds']) >= 2:
        initial_bounds = history['mean_in_bounds'][0]
        final_bounds = history['mean_in_bounds'][-1]
        print(f"  In-bounds ratio: {initial_bounds:.2f} → {final_bounds:.2f}")
else:
    print("  ⚠️  Not enough data points")
    criterion_3 = None

# Final Verdict
print("\n" + "="*60)
print("VERDICT")
print("="*60)

all_pass = criterion_1 and criterion_2 and criterion_3

if all_pass:
    print("\n✅ READY FOR PHASE 4!")
    print("   All criteria passed. Proceed with Surface Strain Regularization.")
elif criterion_1 == False:
    print("\n❌ NOT READY - Training Unstable")
    print("   Recommendations:")
    print("   - Reduce λ_connectivity")
    print("   - Add gradient clipping")
    print("   - Check for numerical issues in Newton iteration")
elif criterion_2 == False or criterion_2 is None:
    print("\n⚠️  NOT READY - Connectivity Not Helping")
    print("   Consider Phase 3.5:")
    print("   - Filter saddles to near-surface ones")
    print("   - Increase cache interval")
    print("   - Clamp Newton steps")
    print("   - Add eikonal strength")
elif criterion_3 == False:
    print("\n❌ NOT READY - Diversity Destroying Validity")
    print("   Recommendations:")
    print("   - Reduce λ_diversity")
    print("   - Add validity constraint")
    print("   - Check diversity loss implementation")
else:
    print("\n⚠️  INCONCLUSIVE - Need longer training run")
    print("   Run for more steps to get clearer signal.")

# Save history
import json
history_serializable = {k: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                            for v in vals] 
                        for k, vals in history.items()}

os.makedirs('scripts/logs', exist_ok=True)
with open('scripts/logs/phase3_readiness.json', 'w') as f:
    json.dump(history_serializable, f, indent=2)
print(f"\n📁 Training history saved to scripts/logs/phase3_readiness.json")

print("\n" + "="*60)
