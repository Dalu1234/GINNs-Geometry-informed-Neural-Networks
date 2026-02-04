"""Validate a GINN config (default: LEGO 1xN). Load config, print key fields, optionally build model."""
import argparse
import sys

sys.path.insert(0, '.')

def main():
    parser = argparse.ArgumentParser(description='Validate GINN config')
    parser.add_argument('yml', nargs='?', default='configs/GINN/lego_1xN_wire.yml', help='Config path (default: LEGO 1xN)')
    parser.add_argument('--no-model', action='store_true', help='Skip model build (only load config)')
    args = parser.parse_args()

    base_yml = 'configs/base_config.yml'
    try:
        from util.get_config import get_config_from_yml
    except ImportError as e:
        print(f"[FAIL] Cannot load config (missing deps?): {e}")
        sys.exit(1)

    try:
        cfg = get_config_from_yml(args.yml, base_yml)
        print("[OK] Config loaded successfully!")
    except Exception as e:
        print(f"[FAIL] Config load error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Common / LEGO-relevant keys
    problem_str = cfg.get('problem', {}).get('problem_str') if isinstance(cfg.get('problem'), dict) else 'N/A'
    print(f"\n  problem_str: {problem_str}")
    print(f"  ginn_bsize: {cfg.get('ginn_bsize', 'N/A')}")
    print(f"  nx: {cfg.get('nx', 'N/A')}, nz: {cfg.get('nz', 'N/A')}")
    print(f"  level_set: {cfg.get('level_set', 'N/A')}")
    print(f"  lambda_if: {cfg.get('lambda_if', 0)}, lambda_env: {cfg.get('lambda_env', 0)}, lambda_eikonal: {cfg.get('lambda_eikonal', 0)}")
    print(f"  lambda_scc: {cfg.get('lambda_scc', 0)}")

    model_cfg = cfg.get('model', {})
    if model_cfg:
        print(f"\n  model_str: {model_cfg.get('model_str', 'N/A')}")
        print(f"  layers: {model_cfg.get('layers', 'N/A')}")
        print(f"  use_tiled_coords: {model_cfg.get('use_tiled_coords', 'N/A')}")
        print(f"  use_dist_to_edge: {model_cfg.get('use_dist_to_edge', 'N/A')}")
        print(f"  use_hypernet: {model_cfg.get('use_hypernet', 'N/A')}")
        if model_cfg.get('use_hypernet'):
            print(f"    lora_rank: {model_cfg.get('lora_rank', 'N/A')}, c_dim: {model_cfg.get('c_dim', 'N/A')}")

    if problem_str == 'lego_1xN':
        print(f"\n  condition_on_n_studs: {cfg.get('condition_on_n_studs', 'N/A')}")
        print(f"  n_studs_values: {cfg.get('n_studs_values', 'N/A')}")
        print(f"  condition_on_n_studs_y: {cfg.get('condition_on_n_studs_y', 'N/A')}")
        print(f"  condition_on_height: {cfg.get('condition_on_height', 'N/A')}")
        print(f"  height_values: {cfg.get('height_values', 'N/A')}")

    if args.no_model:
        print("\n[OK] Validation done (model build skipped).")
        return

    # Build model with same kwargs as trainer (no conditioning increments here; use final nz from config)
    print("\nBuilding model...")
    try:
        import torch
        from util.misc import get_model
        model_kw = dict(cfg['model'])
        model_kw['layers'] = list(model_kw.get('layers', [128, 128, 128]))
        # Apply same nz increments as ginn_trainer for LEGO
        if problem_str == 'lego_1xN':
            if cfg.get('condition_on_n_studs'):
                model_kw['nz'] = model_kw['nz'] + 1
            if cfg.get('condition_on_n_studs_y'):
                model_kw['nz'] = model_kw['nz'] + 1
            if cfg.get('condition_on_height'):
                model_kw['nz'] = model_kw['nz'] + 1
        model = get_model(**model_kw)
        nz = model_kw['nz']
        nx = model_kw['nx']
        # Quick forward
        x = torch.randn(2, nx)
        z = torch.randn(2, nz)
        out = model(x, z)
        assert out.shape == (2, 1), f"expected (2, 1), got {out.shape}"
        print(f"  [OK] Model built and forward (2, {nx}) x (2, {nz}) -> {out.shape}")
        print("[OK] Validation done (config + model). Ready to train.")
    except Exception as e:
        print(f"[FAIL] Model build/forward error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
