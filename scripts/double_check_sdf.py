"""
Minimal script to probe model output at the center of the domain.
This double-checks if the model is outputting shapes (negative SDF) or just empty space.
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from util.get_config import get_config_from_yml
from util.misc import get_model
from util.checkpointing import load_yaml_and_drop_keys
from models.net_w_partials import NetWithPartials

def probe_model(model_path, config_path, use_legacy_gabor=False, label="Model"):
    print(f"\n--- Probing {label} ---")
    print(f"Path: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load bounds first to know where the center is
    bounds_np = np.load('GINN/simJEB/data/bounds.npy')
    bounds = torch.from_numpy(bounds_np).float().to(device)
    center = bounds.mean(dim=1)  # [3]
    print(f"Domain Bounds:\n{bounds_np}")
    print(f"Domain Center: {center.cpu().numpy()}")
    
    # Load config and model
    try:
        config = load_yaml_and_drop_keys(str(config_path), keys_to_drop=[])
        model_kwargs = config['model']
        # Override legacy flag if needed
        model_kwargs['use_legacy_gabor'] = use_legacy_gabor
        
        model = get_model(**model_kwargs)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Wrap with NetWithPartials for easy calling
        netp = NetWithPartials.create_from_model(model, config['nz'], config['nx'])
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Probe 10 random latent vectors at the center point
    print("\nProbing SDF at center point for 10 random z vectors:")
    
    nz = config['nz']
    with torch.no_grad():
        x_probe = center.unsqueeze(0)  # [1, 3]
        
        for i in range(10):
            z_probe = torch.rand(1, nz, device=device) * 2 - 1  # [-1, 1]
            
            # Use netp.f_ direclty: f(params, x, z)
            # Note: netp.f_ is vmapped, so it expects batched inputs?
            # NetWithPartials code: 
            # def f(params, x, z): return functional_call(model, params, (x, z))
            # vf = vmap(f, ...)
            
            # Let's call the model directly to avoid vmap confusion
            val = model(x_probe, z_probe)
            val = val.item()
            
            status = "INSIDE" if val < 0 else "OUTSIDE"
            print(f"  z[{i}]: SDF = {val:.6f} ({status})")

def main():
    # 1. Probe Original Model (Known Good)
    original_model = Path("checkpoints/GINN-model.pt")
    original_config = Path("checkpoints/GINN-config.yml")
    if original_model.exists():
        probe_model(original_model, original_config, use_legacy_gabor=True, label="ORIGINAL GINN")
    
    # 2. Probe New Model
    # Find latest checkpoint
    checkpoints = list(Path("checkpoints").glob("**/*model.pt"))
    # Filter out the original GINN-model.pt
    checkpoints = [p for p in checkpoints if "GINN-model.pt" not in p.name]
    
    if checkpoints:
        latest_model = max(checkpoints, key=lambda p: p.stat().st_mtime)
        # Assuming new model uses current config structure
        config_path = "configs/GINN/simjeb_connectivity_test.yml"
        probe_model(latest_model, config_path, use_legacy_gabor=False, label="NEW CONNECTIVITY MODEL")
    else:
        print("No new checkpoints found.")

if __name__ == "__main__":
    main()
