"""Quick config validation for connectivity test."""
import sys
sys.path.insert(0, '.')

from util.get_config import get_config_from_yml

try:
    cfg = get_config_from_yml('configs/GINN/simjeb_connectivity_test.yml', 'configs/base_config.yml')
    print("✓ Config loaded successfully!")
    print(f"  lambda_connectivity: {cfg.get('lambda_connectivity', 0)}")
    print(f"  training_mode: {cfg.get('training_mode', 'N/A')}")
    print(f"  ginn_bsize: {cfg.get('ginn_bsize', 'N/A')}")
    print(f"  lambda_env: {cfg.get('lambda_env', 0)}")
    print(f"  lambda_if: {cfg.get('lambda_if', 0)}")
    print(f"  lambda_scc: {cfg.get('lambda_scc', 0)}")
    print(f"  connectivity_near_surface_threshold: {cfg.get('connectivity_near_surface_threshold', 'N/A')}")
except Exception as e:
    print(f"✗ Config error: {e}")
    import traceback
    traceback.print_exc()
