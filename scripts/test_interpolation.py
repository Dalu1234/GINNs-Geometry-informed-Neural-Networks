"""
Interpolation Test for Latent Space Quality

This script tests the quality of the latent space by:
1. SMOOTHNESS: Outputs change gradually between z1 and z2
2. VALIDITY: All intermediate shapes satisfy constraints (connected, has volume)
3. NO HOLES: No z values along the path produce garbage

Run: python scripts/test_interpolation.py [--checkpoint PATH] [--n_pairs N] [--n_steps N]
"""

import torch
import numpy as np
from pathlib import Path
import sys
import argparse
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from skimage import measure
from scipy import ndimage
import trimesh
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
# Helper Functions
# ============================================================

def count_mesh_components(vertices, faces):
    """Count connected components in a mesh using trimesh."""
    if len(vertices) == 0 or len(faces) == 0:
        return 0
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    components = mesh.split(only_watertight=False)
    return len(components)


def count_volume_components(sdf_grid, level=0.0):
    """Count connected components in a voxel grid."""
    binary = (sdf_grid < level).astype(np.int32)
    labeled, n_components = ndimage.label(binary)
    return n_components


def extract_mesh_from_sdf(sdf_grid, level=0.0):
    """Extract mesh using marching cubes."""
    try:
        verts, faces, _, _ = measure.marching_cubes(sdf_grid, level=level)
        return verts, faces
    except:
        return np.array([]), np.array([])


def compute_mesh_volume(vertices, faces):
    """Compute volume of mesh."""
    if len(vertices) == 0 or len(faces) == 0:
        return 0.0
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if mesh.is_watertight:
            return abs(mesh.volume)
        else:
            # Approximate with convex hull
            return abs(mesh.convex_hull.volume) * 0.5
    except:
        return 0.0


def interpolate_z(z1, z2, alpha):
    """Linear interpolation between two z vectors."""
    return (1 - alpha) * z1 + alpha * z2


def compute_shape_difference(verts1, faces1, verts2, faces2):
    """
    Compute a difference metric between two shapes.
    Uses Chamfer-like distance on vertices.
    """
    if len(verts1) == 0 or len(verts2) == 0:
        return float('inf')
    
    # Subsample for speed
    n_sample = min(1000, len(verts1), len(verts2))
    idx1 = np.random.choice(len(verts1), n_sample, replace=len(verts1) < n_sample)
    idx2 = np.random.choice(len(verts2), n_sample, replace=len(verts2) < n_sample)
    v1 = verts1[idx1]
    v2 = verts2[idx2]
    
    # Compute average nearest neighbor distance (asymmetric)
    from scipy.spatial import cKDTree
    tree1 = cKDTree(v1)
    tree2 = cKDTree(v2)
    
    d1, _ = tree2.query(v1)  # For each point in v1, distance to nearest in v2
    d2, _ = tree1.query(v2)  # For each point in v2, distance to nearest in v1
    
    # Symmetric Chamfer distance
    chamfer = np.mean(d1) + np.mean(d2)
    return chamfer


# ============================================================
# Core Test Functions
# ============================================================

def generate_mesh_for_z(netp, z, bounds, resolution=64, device='cuda'):
    """Generate mesh for a given z latent vector."""
    from util.visualization.utils_mesh import get_watertight_mesh_for_latent
    
    try:
        verts, faces = get_watertight_mesh_for_latent(
            netp.f_, netp.params, z, bounds,
            mc_resolution=resolution,
            device=device,
            chunks=1,
            level=0,
            surpress_watertight=True
        )
        return verts, faces
    except Exception as e:
        return np.array([]), np.array([])


def test_interpolation_path(netp, z1, z2, bounds, n_steps=10, resolution=64, device='cuda'):
    """
    Test a single interpolation path from z1 to z2.
    
    Returns:
        dict with:
        - alphas: interpolation values
        - n_components: component count at each step
        - volumes: mesh volume at each step
        - valid: boolean indicating if shape exists
        - smoothness_scores: difference between consecutive shapes
        - meshes: list of (verts, faces) tuples
    """
    alphas = np.linspace(0, 1, n_steps)
    results = {
        'alphas': alphas.tolist(),
        'n_components': [],
        'volumes': [],
        'valid': [],
        'smoothness_scores': [],
        'meshes': []
    }
    
    prev_verts, prev_faces = None, None
    
    for alpha in alphas:
        z = interpolate_z(z1, z2, alpha)
        verts, faces = generate_mesh_for_z(netp, z, bounds, resolution, device)
        
        n_comp = count_mesh_components(verts, faces)
        vol = compute_mesh_volume(verts, faces)
        is_valid = len(verts) > 0
        
        results['n_components'].append(n_comp)
        results['volumes'].append(vol)
        results['valid'].append(is_valid)
        results['meshes'].append((verts, faces))
        
        # Compute smoothness (difference from previous shape)
        if prev_verts is not None and is_valid:
            diff = compute_shape_difference(prev_verts, prev_faces, verts, faces)
            results['smoothness_scores'].append(diff)
        else:
            results['smoothness_scores'].append(0.0)
        
        if is_valid:
            prev_verts, prev_faces = verts, faces
    
    return results


def analyze_path_results(results):
    """Analyze interpolation path results and compute metrics."""
    n_steps = len(results['alphas'])
    
    # Validity: fraction of steps with valid shapes
    validity_rate = sum(results['valid']) / n_steps
    
    # Connectivity: fraction of valid shapes that are connected
    valid_comps = [c for c, v in zip(results['n_components'], results['valid']) if v and c > 0]
    connected_rate = sum(1 for c in valid_comps if c == 1) / len(valid_comps) if valid_comps else 0
    
    # Holes: any step with invalid shape (zero volume or no mesh)
    n_holes = sum(1 for v in results['valid'] if not v)
    
    # Smoothness: average and max change between consecutive shapes
    smoothness = results['smoothness_scores'][1:]  # Skip first (always 0)
    avg_smoothness = np.mean(smoothness) if smoothness else float('inf')
    max_smoothness = np.max(smoothness) if smoothness else float('inf')
    
    # Volume consistency: std of volumes (should be low for smooth interpolation)
    valid_volumes = [v for v, is_valid in zip(results['volumes'], results['valid']) if is_valid and v > 0]
    volume_std = np.std(valid_volumes) if valid_volumes else float('inf')
    volume_mean = np.mean(valid_volumes) if valid_volumes else 0
    volume_cv = volume_std / volume_mean if volume_mean > 0 else float('inf')
    
    return {
        'validity_rate': validity_rate,
        'connected_rate': connected_rate,
        'n_holes': n_holes,
        'avg_smoothness': avg_smoothness,
        'max_smoothness': max_smoothness,
        'volume_cv': volume_cv,  # Coefficient of variation
        'volume_mean': volume_mean,
    }


# ============================================================
# Visualization
# ============================================================

def plot_interpolation_path(results, analysis, pair_idx, save_path=None):
    """Plot the interpolation path analysis."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    alphas = results['alphas']
    
    # Row 1: Component count along path
    ax1 = fig.add_subplot(gs[0, :2])
    colors = ['green' if c == 1 else 'red' if c > 1 else 'gray' for c in results['n_components']]
    ax1.bar(alphas, results['n_components'], width=0.08, color=colors, edgecolor='black')
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Target (1 component)')
    ax1.set_xlabel('Interpolation α')
    ax1.set_ylabel('# Components')
    ax1.set_title(f'Connectivity Along Path (Pair {pair_idx+1})')
    ax1.legend()
    
    # Row 1: Volume along path
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.plot(alphas, results['volumes'], 'b-o', linewidth=2, markersize=6)
    ax2.fill_between(alphas, results['volumes'], alpha=0.3)
    ax2.set_xlabel('Interpolation α')
    ax2.set_ylabel('Volume')
    ax2.set_title(f'Volume Along Path (CV={analysis["volume_cv"]:.3f})')
    
    # Row 2: Smoothness (shape change between steps)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(alphas[1:], results['smoothness_scores'][1:], 'orange', linewidth=2, marker='s')
    ax3.axhline(y=analysis['avg_smoothness'], color='red', linestyle='--', label=f'Avg: {analysis["avg_smoothness"]:.4f}')
    ax3.set_xlabel('Interpolation α')
    ax3.set_ylabel('Shape Change (Chamfer)')
    ax3.set_title('Smoothness: Consecutive Shape Difference')
    ax3.legend()
    
    # Row 2: Validity heatmap
    ax4 = fig.add_subplot(gs[1, 2:])
    validity_colors = ['green' if v else 'red' for v in results['valid']]
    ax4.bar(alphas, [1]*len(alphas), width=0.08, color=validity_colors)
    ax4.set_ylim(0, 1.5)
    ax4.set_xlabel('Interpolation α')
    ax4.set_yticks([])
    ax4.set_title(f'Validity: {analysis["validity_rate"]*100:.0f}% valid, {analysis["n_holes"]} holes')
    
    # Row 3: Summary text
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    summary_text = f"""
    INTERPOLATION PATH ANALYSIS (Pair {pair_idx+1})
    ══════════════════════════════════════════════════════════════════════════════
    
    VALIDITY:     {analysis['validity_rate']*100:5.1f}%  shapes have valid meshes
    CONNECTIVITY: {analysis['connected_rate']*100:5.1f}%  valid shapes are connected (1 component)
    HOLES:        {analysis['n_holes']:5d}    steps produced no valid mesh
    
    SMOOTHNESS:
      Average shape change: {analysis['avg_smoothness']:.4f} (lower = smoother)
      Maximum shape change: {analysis['max_smoothness']:.4f}
    
    VOLUME STABILITY:
      Mean volume:  {analysis['volume_mean']:.4f}
      CV (std/mean): {analysis['volume_cv']:.4f} (lower = more stable)
    
    ══════════════════════════════════════════════════════════════════════════════
    PASS CRITERIA:
      ✓ Validity > 90%    : {'PASS ✓' if analysis['validity_rate'] > 0.9 else 'FAIL ✗'}
      ✓ Connectivity > 80%: {'PASS ✓' if analysis['connected_rate'] > 0.8 else 'FAIL ✗'}
      ✓ No holes          : {'PASS ✓' if analysis['n_holes'] == 0 else 'FAIL ✗'}
      ✓ Volume CV < 0.5   : {'PASS ✓' if analysis['volume_cv'] < 0.5 else 'FAIL ✗'}
    """
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
             fontfamily='monospace', verticalalignment='top')
    
    plt.suptitle(f'Latent Space Interpolation Test - Pair {pair_idx+1}', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {save_path}")
    
    return fig


def plot_shape_sequence(results, pair_idx, n_show=5, save_path=None):
    """Plot a sequence of shapes along the interpolation path."""
    n_steps = len(results['meshes'])
    indices = np.linspace(0, n_steps-1, n_show, dtype=int)
    
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4), subplot_kw={'projection': '3d'})
    if n_show == 1:
        axes = [axes]
    
    for ax_idx, step_idx in enumerate(indices):
        verts, faces = results['meshes'][step_idx]
        alpha = results['alphas'][step_idx]
        n_comp = results['n_components'][step_idx]
        
        ax = axes[ax_idx]
        
        if len(verts) > 0 and len(faces) > 0:
            # Subsample faces for visualization
            max_faces = 5000
            if len(faces) > max_faces:
                face_idx = np.random.choice(len(faces), max_faces, replace=False)
                faces_show = faces[face_idx]
            else:
                faces_show = faces
            
            # Color by component count
            color = 'green' if n_comp == 1 else 'red' if n_comp > 1 else 'gray'
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                           triangles=faces_show, alpha=0.7, color=color, edgecolor='none')
        
        ax.set_title(f'α={alpha:.2f}\n{n_comp} comp', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.suptitle(f'Shape Sequence Along Interpolation (Pair {pair_idx+1})', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved shape sequence to {save_path}")
    
    return fig


# ============================================================
# Main Test Runner
# ============================================================

def load_model(checkpoint_path=None, use_original=False):
    """Load model from checkpoint."""
    from util.checkpointing import load_yaml_and_drop_keys
    from util.misc import get_model
    from models.net_w_partials import NetWithPartials
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_original or checkpoint_path is None:
        # Load original GINN model
        config_path = Path("checkpoints/GINN-config.yml")
        model_path = Path("checkpoints/GINN-model.pt")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Original model not found at {model_path}")
        
        config = load_yaml_and_drop_keys(str(config_path), keys_to_drop=[])
        model = get_model(**config['model'], use_legacy_gabor=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded ORIGINAL model from {model_path}")
    else:
        # Load specified checkpoint
        checkpoint_path = Path(checkpoint_path)
        
        # Find corresponding config
        config_path = checkpoint_path.parent / checkpoint_path.name.replace('-model.pt', '-config.yml')
        if not config_path.exists():
            config_path = Path("checkpoints/GINN-config.yml")  # Fallback
        
        config = load_yaml_and_drop_keys(str(config_path), keys_to_drop=[])
        
        # IMPORTANT: For fine-tuned models, use ORIGINAL GINN config for model architecture
        # because the fine-tuned checkpoints inherit the original architecture
        original_config = load_yaml_and_drop_keys("checkpoints/GINN-config.yml", keys_to_drop=[])
        model = get_model(**original_config['model'], use_legacy_gabor=True)
        
        # Load state dict
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded FINE-TUNED model from {checkpoint_path}")
        print(f"  (using original GINN architecture)")
    
    model = model.to(device)
    model.eval()
    
    netp = NetWithPartials.create_from_model(model, config['nz'], config['nx'])
    bounds = torch.from_numpy(np.load('GINN/simJEB/data/bounds.npy')).float().to(device)
    
    return netp, config, bounds, device


def run_interpolation_test(netp, config, bounds, device, n_pairs=5, n_steps=10, 
                           resolution=64, output_dir='scripts/logs/interpolation',
                           z_range=None):
    """Run the full interpolation test."""
    nz = config.get('nz', 8)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("LATENT SPACE INTERPOLATION TEST")
    print("="*70)
    print(f"Testing {n_pairs} random z pairs with {n_steps} interpolation steps each")
    print(f"Latent dimension: {nz}")
    print(f"Mesh resolution: {resolution}")
    print(f"Output directory: {output_dir}")
    print("="*70 + "\n")
    
    # Determine z range from config or use default
    if z_range is None:
        # Try to get from config
        z_interval = config.get('latent_sampling', {}).get('z_sample_interval', [-1, 1])
        z_min, z_max = z_interval[0], z_interval[1]
    else:
        z_min, z_max = z_range
    
    print(f"Using z range: [{z_min}, {z_max}]")
    
    # Generate random z pairs
    torch.manual_seed(42)  # For reproducibility
    z_pairs = []
    for i in range(n_pairs):
        z1 = torch.rand(nz, device=device) * (z_max - z_min) + z_min
        z2 = torch.rand(nz, device=device) * (z_max - z_min) + z_min
        z_pairs.append((z1, z2))
    
    all_analyses = []
    all_results = []
    
    for pair_idx, (z1, z2) in enumerate(tqdm(z_pairs, desc="Testing pairs")):
        print(f"\n--- Pair {pair_idx+1}/{n_pairs} ---")
        print(f"  z1: [{z1[0].item():.3f}, {z1[1].item():.3f}, ...]")
        print(f"  z2: [{z2[0].item():.3f}, {z2[1].item():.3f}, ...]")
        
        # Test interpolation path
        results = test_interpolation_path(netp, z1, z2, bounds, n_steps, resolution, device)
        analysis = analyze_path_results(results)
        
        all_results.append(results)
        all_analyses.append(analysis)
        
        # Print summary
        print(f"  Validity:     {analysis['validity_rate']*100:.1f}%")
        print(f"  Connectivity: {analysis['connected_rate']*100:.1f}%")
        print(f"  Holes:        {analysis['n_holes']}")
        print(f"  Smoothness:   {analysis['avg_smoothness']:.4f} avg, {analysis['max_smoothness']:.4f} max")
        
        # Generate plots
        plot_interpolation_path(
            results, analysis, pair_idx,
            save_path=output_dir / f'interpolation_pair_{pair_idx+1}_analysis.png'
        )
        plot_shape_sequence(
            results, pair_idx, n_show=5,
            save_path=output_dir / f'interpolation_pair_{pair_idx+1}_shapes.png'
        )
        plt.close('all')
    
    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)
    
    avg_validity = np.mean([a['validity_rate'] for a in all_analyses])
    avg_connectivity = np.mean([a['connected_rate'] for a in all_analyses])
    total_holes = sum(a['n_holes'] for a in all_analyses)
    avg_smoothness = np.mean([a['avg_smoothness'] for a in all_analyses])
    max_smoothness = max(a['max_smoothness'] for a in all_analyses)
    avg_volume_cv = np.mean([a['volume_cv'] for a in all_analyses if a['volume_cv'] != float('inf')])
    
    print(f"  Average Validity:     {avg_validity*100:.1f}%")
    print(f"  Average Connectivity: {avg_connectivity*100:.1f}%")
    print(f"  Total Holes:          {total_holes}")
    print(f"  Average Smoothness:   {avg_smoothness:.4f}")
    print(f"  Max Smoothness:       {max_smoothness:.4f}")
    print(f"  Average Volume CV:    {avg_volume_cv:.4f}")
    
    # Pass/Fail summary
    print("\n" + "-"*70)
    print("PASS/FAIL CRITERIA")
    print("-"*70)
    
    pass_validity = avg_validity > 0.9
    pass_connectivity = avg_connectivity > 0.8
    pass_holes = total_holes == 0
    pass_volume = avg_volume_cv < 0.5
    
    print(f"  Validity > 90%:       {'PASS ✓' if pass_validity else 'FAIL ✗'} ({avg_validity*100:.1f}%)")
    print(f"  Connectivity > 80%:   {'PASS ✓' if pass_connectivity else 'FAIL ✗'} ({avg_connectivity*100:.1f}%)")
    print(f"  No holes:             {'PASS ✓' if pass_holes else 'FAIL ✗'} ({total_holes} holes)")
    print(f"  Volume CV < 0.5:      {'PASS ✓' if pass_volume else 'FAIL ✗'} ({avg_volume_cv:.4f})")
    
    overall_pass = bool(pass_validity and pass_connectivity and pass_holes and pass_volume)
    print("\n" + "="*70)
    print(f"OVERALL: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print("="*70)
    
    # Save JSON results (convert numpy types to Python types for JSON)
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_pairs': int(n_pairs),
            'n_steps': int(n_steps),
            'resolution': int(resolution),
            'nz': int(nz),
        },
        'aggregate': {
            'avg_validity': float(avg_validity),
            'avg_connectivity': float(avg_connectivity),
            'total_holes': int(total_holes),
            'avg_smoothness': float(avg_smoothness),
            'max_smoothness': float(max_smoothness),
            'avg_volume_cv': float(avg_volume_cv) if avg_volume_cv != float('inf') else None,
            'overall_pass': overall_pass,
        },
        'per_pair': [
            {
                'pair_idx': i,
                'validity_rate': float(a['validity_rate']),
                'connected_rate': float(a['connected_rate']),
                'n_holes': int(a['n_holes']),
                'avg_smoothness': float(a['avg_smoothness']),
                'max_smoothness': float(a['max_smoothness']),
                'volume_cv': float(a['volume_cv']) if a['volume_cv'] != float('inf') else None,
            }
            for i, a in enumerate(all_analyses)
        ]
    }
    
    json_path = output_dir / 'interpolation_test_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    return json_results


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Test latent space interpolation quality')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: original GINN)')
    parser.add_argument('--original', action='store_true',
                       help='Force use of original GINN model')
    parser.add_argument('--n_pairs', type=int, default=5,
                       help='Number of random z pairs to test (default: 5)')
    parser.add_argument('--n_steps', type=int, default=10,
                       help='Number of interpolation steps per pair (default: 10)')
    parser.add_argument('--resolution', type=int, default=64,
                       help='Mesh resolution for marching cubes (default: 64)')
    parser.add_argument('--output', type=str, default='scripts/logs/interpolation',
                       help='Output directory for results')
    parser.add_argument('--z_min', type=float, default=None,
                       help='Minimum z value (default: from config)')
    parser.add_argument('--z_max', type=float, default=None,
                       help='Maximum z value (default: from config)')
    
    args = parser.parse_args()
    
    # Determine z_range
    z_range = None
    if args.z_min is not None and args.z_max is not None:
        z_range = (args.z_min, args.z_max)
    
    # Load model
    netp, config, bounds, device = load_model(
        checkpoint_path=args.checkpoint,
        use_original=args.original or args.checkpoint is None
    )
    
    # Run test
    results = run_interpolation_test(
        netp, config, bounds, device,
        n_pairs=args.n_pairs,
        n_steps=args.n_steps,
        resolution=args.resolution,
        output_dir=args.output,
        z_range=z_range
    )
    
    return results


if __name__ == "__main__":
    main()
