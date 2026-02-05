import datetime
import glob
import os
import time

import numpy as np

from models.model_factory import ModelFactory
from util.const import MODELS_PARENT_DIR
from util.misc import is_every_n_epochs_fulfilled

import torch
import yaml
from pathlib import Path


def find_model_file(run_id, root_dir, use_epoch=None, get_file='model.pt', suppress_print=False):
    # search in all subdirectories
    start_t = time.time()
    ## /MODELS_PARENT_DIR/cond_siren/2024_05_11__15_55_28-4lsf3jsk/2024_05_11__15_55_28-4lsf3jsk-it_01850-model.pt
    glob_pattern = os.path.join(root_dir, '*', f'*{run_id}', f'*{run_id}*{get_file}')
    candidate_files = glob.glob(glob_pattern, recursive=True)
    
    if len(candidate_files) == 0:
        print(f"No files found in fast search. Note that for this the pattern must be root_dir/SINGLE_DIR_INBETWEEN/*{run_id}/*{run_id}*{get_file}")
        print(f'e.g.: MODELS_PARENT_DIR/cond_siren/2024_05_11__15_55_28-4lsf3jsk/2024_05_11__15_55_28-4lsf3jsk-it_01850-model.pt')
        print(f'Starting slower search...')
        candidate_files = glob.glob(os.path.join(root_dir, '**' f'*{run_id}', f'*{run_id}*{get_file}'), recursive=True)
        print(f'Finished slower search')
        
    print(f'Time to search for candidates: {time.time() - start_t:.2f}')
    print(f"Found {len(candidate_files)} candidate files") 
       
    assert len(candidate_files) > 0, f"No files found for {run_id} in {root_dir}"
    
    if not suppress_print:
        for file in candidate_files:
            print(file)

    if use_epoch is None:
        ## return file from latest epoch
        ## /MODELS_PARENT_DIR/cond_siren/2024_05_11__15_55_28-4lsf3jsk/2024_05_11__15_55_28-4lsf3jsk-it_01850-model.pt
        def get_epoch_from_filename(file):
            spl = file.split('-')
            for s in spl:
                if 'it_' in s:
                    return int(s.split('_')[-1])
        ## sort by epoch
        sorted_by_epoch = sorted(candidate_files, key=get_epoch_from_filename)
        return sorted_by_epoch[-1]

    else:
        for file in candidate_files:
            if f'it_{use_epoch:05}' in file:
                return file



def save_model_every_n_epochs(model, optim, sched, config, epoch, conditional_prior=None):
    if not is_every_n_epochs_fulfilled(epoch, config, 'save_every_n_epochs'):
        return False, None

    assert 'model_save_path' in config, 'model_save_path must be specified in config'

    ## create directory, save config only if it does not exist
    if 'save_model_dir' not in config:
        name_stem = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '-' + config['wandb_id']
        save_subdir = config.get('model_save_subdir', config['model']['model_str'])
        model_parent_path = os.path.join(config['model_save_path'], save_subdir, name_stem) if save_subdir else os.path.join(config['model_save_path'], name_stem)
        config['save_model_dir'] = model_parent_path

        ## create parent directory if it does not exist
        if not os.path.exists(model_parent_path):
            Path(model_parent_path).mkdir(parents=True, exist_ok=True)
            
        # save config as yml
        config_filename = name_stem + '-config.yml'
        config_path = os.path.join(model_parent_path, config_filename)
        # remove all keys that are tensors; important for loading later
        config = {k: v for k, v in config.items() if not (isinstance(v, torch.Tensor) or isinstance(v, np.ndarray))}
        if not os.path.exists(config_path):
            with open(config_path, 'w') as file:
                yaml.dump(config, file)

    model_parent_path = config['save_model_dir']
    name_stem = os.path.basename(model_parent_path)  # Cross-platform path handling

    ## add epoch to filename if not overwriting
    if not config['overwrite_existing_saved_model']:
        name_stem += f'-it_{epoch:05}'

    ## save model
    model_filename = name_stem + '-model.pt'
    model_path = os.path.join(model_parent_path, model_filename)
    # torch.save(model.state_dict(), model_path)
    ModelFactory.save(model, model_path)

    ## save conditional prior (e.g. p(z_base|c)) in same dir so resume restores it
    if conditional_prior is not None:
        prior_filename = name_stem + '-prior.pt'
        prior_path = os.path.join(model_parent_path, prior_filename)
        torch.save(conditional_prior.state_dict(), prior_path)

    ## save optimizer
    if config.get('save_optimizer', False):
        optim_filename = name_stem + '-optim.pt'
        optim_path = os.path.join(model_parent_path, optim_filename)
        torch.save(optim.state_dict(), optim_path)

        if config.get('use_scheduler', False):
            ## save scheduler (if used)
            sched_filename = name_stem + '-sched.pt'
            sched_path = os.path.join(model_parent_path, sched_filename)
            torch.save(sched.state_dict(), sched_path)

    return True, model_path


def get_model_path_via_wandb_id_from_fs(run_id, root_dir, use_epoch=None, get_file='model.pt', suppress_print=False):

    try:
        start_t = time.time()
        file_path = find_model_file(run_id, root_dir, use_epoch=use_epoch, get_file=get_file, suppress_print=suppress_print)
        print(f'Found file in {time.time() - start_t:.2f}s at {file_path}')
        return file_path
    except Exception as e:
        print(e)
    raise ValueError(f"Could not find model with {run_id=} anywhere")


def _get_latest_model_path(search_dir):
    """Glob *-model.pt under search_dir, sort by mtime, return path to latest."""
    pattern = os.path.join(search_dir, '**', '*-model.pt')
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    return latest


def _get_model_load_path(config):
    """Return the path used to load the model (for prior load)."""
    if config.get('model_load_latest', False):
        search_dir = config.get('model_load_latest_dir')
        if not search_dir and 'model_save_path' in config and 'model' in config:
            save_subdir = config.get('model_save_subdir', config['model'].get('model_str', ''))
            if save_subdir:
                search_dir = os.path.join(config['model_save_path'], save_subdir)
        if not search_dir:
            raise ValueError('model_load_latest is True but model_load_latest_dir is not set and cannot derive from model_save_path/model_str')
        path = _get_latest_model_path(search_dir)
        if path is None:
            raise ValueError(f'model_load_latest is True but no *-model.pt found under {search_dir}')
        print(f'Loading latest checkpoint: {path}')
        return path
    if 'model_load_path' in config:
        return config['model_load_path']
    if 'model_load_wandb_id' in config:
        return get_model_path_via_wandb_id_from_fs(config['model_load_wandb_id'], root_dir=MODELS_PARENT_DIR, suppress_print=True)
    return None


def load_conditional_prior(config, conditional_prior, device=None):
    """
    If we loaded a model from a checkpoint and conditional_prior exists,
    load prior state from same run (name_stem + '-prior.pt'). No-op if prior is None or file missing.
    """
    if conditional_prior is None:
        return conditional_prior
    if not (config.get('load_model', False) or config.get('load_mos', False)):
        return conditional_prior
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_load_path = _get_model_load_path(config)
    if model_load_path is None:
        return conditional_prior
    prior_path = model_load_path.replace('-model.pt', '-prior.pt')
    if not os.path.isfile(prior_path):
        return conditional_prior
    state = torch.load(prior_path, map_location=device)
    conditional_prior.load_state_dict(state)
    print(f'Loaded conditional prior from {prior_path}')
    return conditional_prior


def load_model_optim_sched(config, model, optim, sched, device=None):
    ## Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ## Load model weights
    if not (config.get('load_model', False) or config.get('load_mos', False)):
        if config.get('model_load_latest') or 'model_load_path' in config or 'model_load_wandb_id' in config:
            print('WARNING: model load option set but load_model/load_mos is False. Ignoring.')
        return model, optim, sched

    if config.get('model_load_latest') and 'model_load_path' in config:
        raise ValueError('model_load_latest and model_load_path cannot be specified at the same time')
    if config.get('model_load_latest') and 'model_load_wandb_id' in config:
        raise ValueError('model_load_latest and model_load_wandb_id cannot be specified at the same time')
    if 'model_load_path' in config and 'model_load_wandb_id' in config:
        raise ValueError('model_load_path and model_load_wandb_id cannot be specified at the same time')
    if not (config.get('model_load_latest') or 'model_load_path' in config or 'model_load_wandb_id' in config):
        raise ValueError('One of model_load_latest, model_load_path or model_load_wandb_id must be specified if load_model/load_mos is True')

    model_load_path = _get_model_load_path(config)
    if model_load_path is None:
        raise ValueError('Could not resolve model load path')

    print(f'Loading model from {model_load_path}...')
    ckpt = torch.load(model_load_path, map_location=device)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)

    if config.get('load_optimizer', False) or config.get('load_mos', False):
        optim_load_path = model_load_path.replace('-model.pt', '-optim.pt')
        if os.path.isfile(optim_load_path):
            try:
                print(f'Loading optimizer from {optim_load_path}...')
                optim.load_state_dict(torch.load(optim_load_path, map_location=device))
            except (ValueError, RuntimeError) as e:
                if 'parameter group' in str(e).lower() or 'size' in str(e).lower():
                    print(f'WARNING: Could not load optimizer (param groups do not match): {e}')
                    print('         Continuing with fresh optimizer; model weights were loaded.')
                else:
                    raise
        else:
            print(f'No optimizer checkpoint at {optim_load_path}, skipping.')

    if config.get('use_scheduler', False) or config.get('load_mos', False):
        sched_load_path = model_load_path.replace('-model.pt', '-sched.pt')
        if os.path.isfile(sched_load_path):
            try:
                print(f'Loading scheduler from {sched_load_path}...')
                sched.load_state_dict(torch.load(sched_load_path, map_location=device))
            except (ValueError, RuntimeError) as e:
                print(f'WARNING: Could not load scheduler: {e}. Continuing with fresh scheduler.')
        else:
            print(f'No scheduler checkpoint at {sched_load_path}, skipping.')

    ## Optionally save back into the same run directory (true "resume current run")
    if config.get('resume_same_run_dir', False):
        config['save_model_dir'] = os.path.dirname(model_load_path)
        print(f'Resuming same run: saves will go to {config["save_model_dir"]}')

    return model, optim, sched


def load_yaml_and_drop_keys(file_path, keys_to_drop):
    def remove_key_from_yaml(yaml_text, key_to_remove):
        lines = yaml_text.split('\n')
        modified_lines = []
        skip = False

        for line in lines:
            stripped_line = line
            if stripped_line.startswith(key_to_remove + ':'):
                skip = True
            elif skip and stripped_line.startswith('-'):
                continue
            elif skip and stripped_line.startswith('  '):
                continue
            else:
                skip = False
                modified_lines.append(line)

        return '\n'.join(modified_lines)

    # Read the YAML file as text
    with open(file_path, 'r') as file:
        yaml_text = file.read()
    for key_to_remove in keys_to_drop:
        yaml_text = remove_key_from_yaml(yaml_text, key_to_remove)
    config = yaml.safe_load(yaml_text)

    return config