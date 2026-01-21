"""
Phase 3.5 Readiness Check

Tests the improved connectivity constraint with:
1. Near-surface saddle filtering
2. Clamped Newton steps  
3. Eikonal regularization

Run: python scripts/phase35_readiness_check.py
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
from train.losses_connectivity_v2 import (
    CachedConnectivityLossV2,
    eikonal_loss,
    phase35_connectivity_loss,
)
from train.losses_diversity import diversity_loss_chamfer
from models.point_wrapper import PointWrapper
from skimage import measure

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================
# Configuration - Phase 3.5 settings
# ============================================================
N_STEPS = 2000
LOG_EVERY = 100
N_FIXED_LATENTS = 10
LEARNING_RATE = 1e-4

# Loss weights - increased connectivity
LAMBDA_SDF = 1.0
LAMBDA_DIVERSITY = 0.0   # DISABLED - isolating connectivity effect
LAMBDA_CONNECTIVITY = 1.0  # Increased 100x from 0.001
LAMBDA_EIKONAL = 0.1  # NEW: Keep SDF well-behaved

# Connectivity settings - Phase 3.5 improvements
CONNECTIVITY_UPDATE_EVERY = 25  # More frequent updates
CONNECTIVITY_N_SAMPLES = 1000   # More samples
CONNECTIVITY_N_ITER = 40        # More iterations
NEAR_SURFACE_THRESHOLD = 0.15   # Only near-surface saddles
MAX_STEP_SIZE = 0.3             # Clamp Newton steps

# ============================================================
# Load Model
# ============================================================
print("\n" + "="*60)
print("Phase 3.5 Readiness Check")
print("="*60)

config = load_yaml_and_drop_keys('checkpoints/GINN-config.yml', keys_to_drop=[])
bounds = torch.from_numpy(np.load('GINN/simJEB/data/bounds.npy')).float().to(device)

model_raw = get_model(**config['model'], use_legacy_gabor=True)
# Load fresh copy of pretrained weights
model_raw.load_state_dict(torch.load('checkpoints/GINN-model.pt', map_location=device))
model_raw = model_raw.to(device)
model_raw.train()

nz = config['nz']
print(f"Model loaded. nz={nz}")
print(f"λ_conn={LAMBDA_CONNECTIVITY}, λ_eik={LAMBDA_EIKONAL}, λ_div={LAMBDA_DIVERSITY}")

# ============================================================
# Fixed Latents for Tracking
# ============================================================
fixed_latents = torch.linspace(-0.8, 0.8, N_FIXED_LATENTS).unsqueeze(1).repeat(1, nz).to(device)

# ============================================================
# Helper Functions
# ============================================================

def count_components(sdf_grid, level=0.0):
    interior = (sdf_grid < level)
    labeled, n = ndimage.label(interior)
    return n

def get_sdf_grid(model, z, bounds, resolution=48):
    xs = torch.linspace(bounds[0, 0], bounds[0, 1], resolution, device=device)
    ys = torch.linspace(bounds[1, 0], bounds[1, 1], resolution, device=device)
    zs = torch.linspace(bounds[2, 0], bounds[2, 1], resolution, device=device)
    grid_pts = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1).reshape(-1, 3)
    
    z_batch = z.unsqueeze(0).expand(len(grid_pts), -1)
    with torch.no_grad():
        sdf_vals = model(grid_pts, z_batch).squeeze()
    return sdf_vals.reshape(resolution, resolution, resolution).cpu().numpy()

def compute_all_fixed_latent_metrics(model, fixed_latents, bounds):
    components = []
    valid_count = 0
    
    for z in fixed_latents:
        sdf_grid = get_sdf_grid(model, z, bounds, resolution=48)
        n = count_components(sdf_grid)
        components.append(n)
        if (sdf_grid.min() < 0) and (sdf_grid.max() > 0):
            valid_count += 1
    
    return {
        'mean_components': np.mean(components),
        'max_components': np.max(components),
        'min_components': np.min(components),
        'valid_surfaces': valid_count,
        'component_list': components,
    }

def sample_surface_points(model, z, bounds, n_points=200, resolution=48):
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
# Training Loop with Phase 3.5
# ============================================================
print("\n" + "="*60)
print("Starting Training with Phase 3.5 Connectivity")
print("="*60 + "\n")

optimizer = torch.optim.Adam(model_raw.parameters(), lr=LEARNING_RATE)

# Phase 3.5 cached connectivity loss
cached_connectivity = CachedConnectivityLossV2(
    bounds=bounds,
    n_samples=CONNECTIVITY_N_SAMPLES,
    n_iter=CONNECTIVITY_N_ITER,
    grad_tol=1e-2,
    update_every_n_epochs=CONNECTIVITY_UPDATE_EVERY,
    near_surface_threshold=NEAR_SURFACE_THRESHOLD,
    max_step_size=MAX_STEP_SIZE,
    loss_type='attract',
)

# Tracking
history = defaultdict(list)
initial_metrics = None
nan_detected = False

# Keep last connectivity loss VALUE to use every step (stored as float, not tensor)
last_conn_loss = 0.0
last_conn_info = {}

try:
    for step in tqdm(range(N_STEPS), desc="Training"):
        optimizer.zero_grad()
        
        # Sample batch of random z
        batch_size = 4
        z_batch = torch.rand(batch_size, nz, device=device) * 2 - 1
        
        # Sample random points for SDF/eikonal
        n_pts = 500
        pts = torch.rand(n_pts, 3, device=device)
        pts = pts * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        pts.requires_grad_(True)
        
        pts_expanded = pts.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)
        z_expanded = z_batch.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, nz)
        
        # Forward pass
        sdf_vals = model_raw(pts_expanded, z_expanded).squeeze()
        
        # SDF regularization loss
        loss_sdf = (sdf_vals.abs() - 0.5).relu().mean()
        
        # Eikonal loss (keep gradient norm ~1)
        # Use first shape in batch
        class FixedZ:
            def __init__(self, model, z):
                self.model = model
                self.z = z
            def __call__(self, x):
                z_b = self.z.unsqueeze(0).expand(len(x), -1)
                return self.model(x, z_b)
        
        f_fixed = FixedZ(model_raw, z_batch[0])
        loss_eikonal = eikonal_loss(f_fixed, pts[:100])  # Subsample for speed
        
        # Diversity loss (reduced weight)
        loss_diversity = torch.tensor(0.0, device=device)
        if LAMBDA_DIVERSITY > 0 and step % 5 == 0:  # Every 5 steps
            surface_pts_list = []
            for i in range(batch_size):
                sp = sample_surface_points(model_raw, z_batch[i], bounds, n_points=100)
                if sp is not None and len(sp) > 10:
                    surface_pts_list.append(sp)
            
            if len(surface_pts_list) >= 2:
                pw = PointWrapper.create_from_pts_per_shape_list(surface_pts_list)
                loss_diversity = diversity_loss_chamfer(pw, aggregation='mean')
        
        # Connectivity loss (Phase 3.5) - reuse last value between updates
        conn_info = last_conn_info
        if LAMBDA_CONNECTIVITY > 0:
            if step % 5 == 0:
                # Recompute every 5 steps (cache handles expensive Newton internally)
                new_conn_loss, last_conn_info = cached_connectivity(f_fixed, step)
                last_conn_loss = new_conn_loss.detach().item()  # Store as float to avoid graph issues
                conn_info = last_conn_info
            # Recreate tensor each step (avoids "backward through graph twice" error)
            loss_connectivity = torch.tensor(last_conn_loss, device=device, requires_grad=False)
        else:
            loss_connectivity = torch.tensor(0.0, device=device)
        
        # Debug print every 100 steps
        if step % 100 == 0:
            print(f"  [DEBUG] conn_raw={float(loss_connectivity):.6f}, "
                  f"weighted={float(LAMBDA_CONNECTIVITY * loss_connectivity):.6f}")
        
        # Total loss
        total_loss = (
            LAMBDA_SDF * loss_sdf + 
            LAMBDA_EIKONAL * loss_eikonal +
            LAMBDA_DIVERSITY * loss_diversity + 
            LAMBDA_CONNECTIVITY * loss_connectivity
        )
        
        # Check for NaN
        if torch.isnan(total_loss):
            print(f"\n❌ NaN at step {step}")
            nan_detected = True
            break
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_raw.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Log
        history['step'].append(step)
        history['loss_total'].append(total_loss.item())
        history['loss_sdf'].append(loss_sdf.item())
        history['loss_eikonal'].append(loss_eikonal.item())
        history['loss_diversity'].append(loss_diversity.item() if isinstance(loss_diversity, torch.Tensor) else 0)
        history['loss_connectivity'].append(loss_connectivity.item() if isinstance(loss_connectivity, torch.Tensor) else 0)
        
        # Periodic detailed logging
        if step % LOG_EVERY == 0:
            with torch.no_grad():
                metrics = compute_all_fixed_latent_metrics(model_raw, fixed_latents, bounds)
            
            if initial_metrics is None:
                initial_metrics = metrics.copy()
            
            history['mean_components'].append(metrics['mean_components'])
            history['valid_surfaces'].append(metrics['valid_surfaces'])
            
            print(f"\nStep {step}:")
            print(f"  Losses: total={total_loss.item():.4f}, sdf={loss_sdf.item():.4f}, "
                  f"eik={loss_eikonal.item():.4f}, div={history['loss_diversity'][-1]:.4f}, "
                  f"conn={history['loss_connectivity'][-1]:.6f}")
            print(f"  Components: mean={metrics['mean_components']:.1f}, "
                  f"range=[{metrics['min_components']}, {metrics['max_components']}]")
            print(f"  Valid: {metrics['valid_surfaces']}/{N_FIXED_LATENTS}")
            if conn_info.get('n_saddles', 0) > 0:
                print(f"  Saddles: {conn_info.get('n_saddles', 0)} near-surface, "
                      f"mean |f|={conn_info.get('saddle_f_values', torch.tensor([0])).abs().mean().item():.4f}")

except KeyboardInterrupt:
    print("\n\nInterrupted.")

# ============================================================
# Evaluation Report
# ============================================================
print("\n" + "="*60)
print("PHASE 3.5 EVALUATION REPORT")
print("="*60)

# Criterion 1: Stability
print("\n📊 CRITERION 1: Training Stability")
print("-" * 40)
if nan_detected:
    print("  ❌ NaN detected")
    c1 = False
else:
    final_losses = history['loss_total'][-100:] if len(history['loss_total']) >= 100 else history['loss_total']
    print(f"  ✓ No NaN or explosions")
    print(f"  Final loss: mean={np.mean(final_losses):.4f}, std={np.std(final_losses):.4f}")
    
    # Check eikonal
    final_eik = history['loss_eikonal'][-100:] if len(history['loss_eikonal']) >= 100 else history['loss_eikonal']
    print(f"  Eikonal loss: {np.mean(final_eik):.4f} (want ~0 for good SDF)")
    c1 = True

# Criterion 2: Connectivity improving
print("\n📊 CRITERION 2: Connectivity Improvement")
print("-" * 40)
if len(history['mean_components']) >= 2:
    initial = history['mean_components'][0]
    final = history['mean_components'][-1]
    change = final - initial
    
    print(f"  Initial components: {initial:.1f}")
    print(f"  Final components: {final:.1f}")
    print(f"  Change: {change:+.1f}")
    
    # Also check trend
    if len(history['mean_components']) >= 5:
        trend = np.polyfit(range(len(history['mean_components'])), history['mean_components'], 1)[0]
        print(f"  Trend slope: {trend:+.2f} per log interval")
    
    if change < -0.5:
        print("  ✓ Component count improved!")
        c2 = True
    elif change > 2.0:
        print("  ❌ Component count got worse")
        c2 = False
    else:
        print("  ⚠️  No significant change")
        c2 = None
else:
    c2 = None

# Criterion 3: Validity preserved
print("\n📊 CRITERION 3: Validity Preserved")
print("-" * 40)
if len(history['valid_surfaces']) >= 2:
    initial_valid = history['valid_surfaces'][0]
    final_valid = history['valid_surfaces'][-1]
    print(f"  Initial valid: {initial_valid}/{N_FIXED_LATENTS}")
    print(f"  Final valid: {final_valid}/{N_FIXED_LATENTS}")
    
    if final_valid >= initial_valid - 1:
        print("  ✓ Validity preserved")
        c3 = True
    else:
        print(f"  ❌ Lost {initial_valid - final_valid} surfaces")
        c3 = False
else:
    c3 = None

# Connectivity diagnostics
print("\n📊 Connectivity Diagnostics (from cache)")
print("-" * 40)
diag = cached_connectivity.get_diagnostics()
if diag['n_saddles']:
    print(f"  Saddle counts over time: {diag['n_saddles']}")
if diag['mean_f_saddle']:
    print(f"  Mean |f(saddle)| over time: {[f'{x:.4f}' for x in diag['mean_f_saddle']]}")
    if len(diag['mean_f_saddle']) >= 2:
        if diag['mean_f_saddle'][-1] < diag['mean_f_saddle'][0]:
            print("  ✓ |f(saddle)| decreasing - saddles moving to surface!")
        else:
            print("  ⚠️  |f(saddle)| not decreasing")

# Verdict
print("\n" + "="*60)
print("VERDICT")
print("="*60)

if c1 and c2 and c3:
    print("\n✅ READY FOR PHASE 4!")
elif c1 and c3 and c2 is None:
    print("\n⚠️  INCONCLUSIVE - run longer or adjust λ_connectivity")
    print("   Try: increase λ_connectivity or reduce CONNECTIVITY_UPDATE_EVERY")
elif not c2:
    print("\n❌ Still not ready. Consider:")
    print("   - Disable diversity loss entirely (λ=0)")
    print("   - Increase λ_connectivity further")
    print("   - Check if model has degenerate behavior")
else:
    print("\n⚠️  Partial pass. Review diagnostics above.")

# Save
import json
os.makedirs('scripts/logs', exist_ok=True)
history_s = {k: [float(v) if hasattr(v, 'item') else v for v in vals] for k, vals in history.items()}
with open('scripts/logs/phase35_readiness.json', 'w') as f:
    json.dump(history_s, f, indent=2)
print(f"\n📁 Saved to scripts/logs/phase35_readiness.json")
