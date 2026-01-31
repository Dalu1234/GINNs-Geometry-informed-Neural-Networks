"""
Test script to verify if trained shapes are actually connected.
Loads the trained model and counts connected components for multiple z samples.
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from skimage import measure
from scipy import ndimage
import trimesh


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


def main():
    # Find the latest checkpoint
    checkpoint_dir = Path("checkpoints")
    
    # Look for our trained model
    model_files = list(checkpoint_dir.glob("**/simjeb_connectivity*model*.pt"))
    if not model_files:
        model_files = list(checkpoint_dir.glob("**/*model*.pt"))
    
    if not model_files:
        print("No checkpoint found! Looking in wandb...")
        # Try wandb directory
        wandb_dir = Path("wandb")
        model_files = list(wandb_dir.glob("**/model*.pt")) + list(wandb_dir.glob("**/*model*.pt"))
    
    if not model_files:
        print("No model checkpoint found.")
        return
    
    print(f"Found checkpoints: {model_files}")
    
    # Use the latest connectivity test checkpoint (by modification time)
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"\nUsing latest: {latest_model}")
    
    # Also test against the original GINN model for comparison
    original_model = Path("checkpoints/GINN-model.pt")
    if original_model.exists():
        print(f"Also testing original: {original_model}")
        test_original_checkpoint(original_model)
    
    # Load and test latest
    test_checkpoint(latest_model)


def test_original_checkpoint(model_path):
    """Test the original pre-trained GINN model (uses legacy_gabor)."""
    from util.checkpointing import load_yaml_and_drop_keys
    from util.misc import get_model
    from models.net_w_partials import NetWithPartials
    
    print("\n" + "="*60)
    print("ORIGINAL GINN MODEL (for comparison)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load original config
    config = load_yaml_and_drop_keys("checkpoints/GINN-config.yml", keys_to_drop=[])
    
    # Original model uses legacy_gabor=True
    model = get_model(**config['model'], use_legacy_gabor=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    netp = NetWithPartials.create_from_model(model, config['nz'], config['nx'])
    bounds = torch.from_numpy(np.load('GINN/simJEB/data/bounds.npy')).float().to(device)
    
    print(f"Original model config: nz={config['nz']}, nx={config['nx']}")
    test_model_connectivity_v2(netp, config, device, bounds)


def test_checkpoint(model_path):
    """Load a specific checkpoint and test connectivity."""
    from util.get_config import get_config_from_yml
    from util.misc import get_model
    from models.net_w_partials import NetWithPartials
    from util.visualization.utils_mesh import get_watertight_mesh_for_latent
    
    print("\n" + "="*60)
    print("CONNECTIVITY TEST - Direct Mesh Analysis")
    print("="*60)
    
    # Determine which config to use based on model path
    model_path_str = str(model_path)
    
    # Models that were fine-tuned FROM original GINN (use_legacy_gabor=True, nz=2)
    # A/B test IDs and SCC fine-tuned IDs
    legacy_gabor_ids = [
        "6sg6q6lo", "trwen22e", "3eynf9bj",  # A/B test runs
        "m8eyb37q", "bnsrxi2p",               # SCC fine-tuned runs (1000ep, 3000ep)
        "qm3mo2zu", "6xaiis7n",               # Other SCC fine-tuned runs
    ]
    
    if any(rid in model_path_str for rid in legacy_gabor_ids) or "AB_test" in model_path_str:
        # Models fine-tuned from original GINN - use original config with legacy_gabor
        config_path = "configs/GINN/simjeb_connectivity_AB_test.yml"
    else:
        # Original connectivity test with nz=8 and new architecture
        config_path = "configs/GINN/simjeb_connectivity_test.yml"
    
    print(f"Using config: {config_path}")
    config = get_config_from_yml(config_path, "configs/base_config.yml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model using get_model from util.misc
    model_cfg = config['model']
    model = get_model(
        model_str=model_cfg['model_str'],
        nx=model_cfg['nx'],
        nz=model_cfg['nz'],
        layers=model_cfg['layers'].copy(),  # copy since get_model modifies it
        w0_initial=model_cfg.get('w0_initial', 30),
        w0=model_cfg.get('w0', 30),
        wire_scale=model_cfg.get('wire_scale', 10),
        return_density=model_cfg.get('return_density', False),
        use_legacy_gabor=model_cfg.get('use_legacy_gabor', False)  # Match config
    )
    model = model.to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")
    
    # Wrap in NetWithPartials like the notebook does
    netp = NetWithPartials.create_from_model(model, config['nz'], config['nx'])
    
    # Load bounds
    bounds = torch.from_numpy(np.load('GINN/simJEB/data/bounds.npy')).float().to(device)
    print(f"Bounds: {bounds}")
    
    # Test connectivity using the proper mesh extraction
    test_model_connectivity_v2(netp, config, device, bounds)


def test_model_connectivity_v2(netp, config, device, bounds):
    """Generate shapes using proper mesh extraction and count components."""
    from util.visualization.utils_mesh import get_watertight_mesh_for_latent
    
    print("\n" + "-"*60)
    print("Generating 20 random shapes and counting components...")
    print("-"*60)
    
    nz = config.get('nz', 8)
    mc_resolution = 64  # Lower for faster testing
    
    results = []
    
    for i in range(20):
        # Random z vector in [-1, 1]
        z = torch.rand(nz, device=device) * 2 - 1
        
        try:
            # Use the same mesh extraction as the notebook
            verts, faces = get_watertight_mesh_for_latent(
                netp.f_, netp.params, z, bounds, 
                mc_resolution=mc_resolution, 
                device=device,
                chunks=1, 
                level=0, 
                surpress_watertight=True
            )
            
            n_mesh = count_mesh_components(verts, faces)
            has_mesh = len(verts) > 0
            
        except Exception as e:
            if i < 3:
                print(f"    [DEBUG] Shape {i+1} error: {e}")
            verts = np.array([])
            faces = np.array([])
            n_mesh = 0
            has_mesh = False
        
        results.append({
            'z_idx': i,
            'mesh_components': n_mesh,
            'has_mesh': has_mesh,
            'n_vertices': len(verts)
        })
        
        status = "✓" if n_mesh == 1 else "✗" if n_mesh > 1 else "∅"
        print(f"  Shape {i+1:2d}: {status} mesh={n_mesh}, vertices={len(verts)}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    valid_shapes = [r for r in results if r['has_mesh']]
    connected_shapes = [r for r in valid_shapes if r['mesh_components'] == 1]
    
    print(f"Total shapes tested:    {len(results)}")
    print(f"Valid shapes (has mesh): {len(valid_shapes)}")
    print(f"Connected (1 component): {len(connected_shapes)}")
    print(f"Disconnected (>1 comp):  {len(valid_shapes) - len(connected_shapes)}")
    
    if valid_shapes:
        avg_components = np.mean([r['mesh_components'] for r in valid_shapes])
        print(f"\nAverage components per valid shape: {avg_components:.2f}")
        
        connectivity_rate = len(connected_shapes) / len(valid_shapes) * 100
        print(f"Connectivity rate: {connectivity_rate:.1f}%")
    
    print("\n" + "="*60)


def test_fresh_evaluation():
    """
    Load config and model, generate shapes, count components.
    """
    from util.get_config import get_config
    from models.model_factory import ModelFactory
    
    print("\n" + "="*60)
    print("CONNECTIVITY TEST - Direct Mesh Analysis")
    print("="*60)
    
    # Load config
    config = get_config("GINN/simjeb_connectivity_test")
    
    # Try to load model from wandb latest run
    import glob
    wandb_runs = sorted(glob.glob("wandb/run-*"))
    
    if wandb_runs:
        latest_run = wandb_runs[-1]
        model_path = Path(latest_run) / "files" / "model.pt"
        if not model_path.exists():
            # Try other locations
            possible_paths = list(Path(latest_run).glob("**/*.pt"))
            if possible_paths:
                model_path = possible_paths[0]
            else:
                print(f"No .pt file in {latest_run}")
                return
        
        print(f"\nLoading model from: {model_path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        model = ModelFactory.create_model(config)
        model = model.to(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("Model loaded successfully!")
        
        # Test connectivity
        test_model_connectivity(model, config, device)
    else:
        print("No wandb runs found.")
        # Check checkpoints folder
        check_path = Path("checkpoints")
        pt_files = list(check_path.glob("*.pt"))
        print(f"Checkpoint files: {pt_files}")


def test_model_connectivity(model, config, device):
    """Generate multiple shapes and count components."""
    
    print("\n" + "-"*60)
    print("Generating 20 random shapes and counting components...")
    print("-"*60)
    
    nz = config.get('nz', 8)
    resolution = 64  # Grid resolution for marching cubes
    
    # Load actual SimJEB bounds
    bounds = np.load('GINN/simJEB/data/bounds.npy')
    print(f"SimJEB bounds: {bounds}")
    
    x_range = torch.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y_range = torch.linspace(bounds[1, 0], bounds[1, 1], resolution)
    z_range = torch.linspace(bounds[2, 0], bounds[2, 1], resolution)
    
    results = []
    
    for i in range(20):
        # Random z vector
        z = torch.rand(1, nz, device=device) * 2 - 1  # [-1, 1]
        
        # Create grid in SimJEB coordinate space
        grid_x, grid_y, grid_z = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
        points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
        points = points.to(device)
        
        # Evaluate model
        with torch.no_grad():
            # Expand z for all points - model takes (x, z) separately
            z_expanded = z.expand(points.shape[0], -1)
            sdf = model(points, z_expanded)
            
            if isinstance(sdf, tuple):
                sdf = sdf[0]
            sdf_vals = sdf.squeeze().cpu().numpy()
        
        # Debug: print SDF range for first few shapes
        if i < 3:
            print(f"    [DEBUG] Shape {i+1} SDF range: [{sdf_vals.min():.3f}, {sdf_vals.max():.3f}]")
        
        # Reshape to grid
        sdf_grid = sdf_vals.reshape(resolution, resolution, resolution)
        
        # Count components in volume
        n_volume = count_volume_components(sdf_grid, level=0.0)
        
        # Count components in mesh
        verts, faces = extract_mesh_from_sdf(sdf_grid, level=0.0)
        n_mesh = count_mesh_components(verts, faces)
        
        results.append({
            'z_idx': i,
            'volume_components': n_volume,
            'mesh_components': n_mesh,
            'has_mesh': len(verts) > 0
        })
        
        status = "✓" if n_mesh == 1 else "✗" if n_mesh > 1 else "∅"
        print(f"  Shape {i+1:2d}: {status} volume={n_volume}, mesh={n_mesh}, vertices={len(verts)}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    valid_shapes = [r for r in results if r['has_mesh']]
    connected_shapes = [r for r in valid_shapes if r['mesh_components'] == 1]
    
    print(f"Total shapes tested:    {len(results)}")
    print(f"Valid shapes (has mesh): {len(valid_shapes)}")
    print(f"Connected (1 component): {len(connected_shapes)}")
    print(f"Disconnected (>1 comp):  {len(valid_shapes) - len(connected_shapes)}")
    
    if valid_shapes:
        avg_components = np.mean([r['mesh_components'] for r in valid_shapes])
        print(f"\nAverage components per valid shape: {avg_components:.2f}")
        
        connectivity_rate = len(connected_shapes) / len(valid_shapes) * 100
        print(f"Connectivity rate: {connectivity_rate:.1f}%")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
