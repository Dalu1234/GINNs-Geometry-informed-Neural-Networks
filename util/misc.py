import os
import random

import numpy as np
import torch


def encode_n_gaussian(N, support_points, sigma=1.0, device=None, dtype=None):
    """
    Encode scalar N as a vector of Gaussian weights over support points (e.g. n_studs_values).
    Unseen N (e.g. 7) gets a blend of nearby training N (6, 8). Weights sum to 1.
    Returns: tensor of shape (len(support_points),); same device/dtype as support if tensor, else optional device/dtype.
    """
    support = torch.as_tensor(support_points, device=device, dtype=dtype or torch.get_default_dtype())
    n_val = N if isinstance(N, (int, float)) else (N.item() if hasattr(N, 'item') else float(N))
    n_val = torch.tensor(n_val, device=support.device, dtype=support.dtype)
    d = (n_val - support) / max(float(sigma), 1e-8)
    w = torch.exp(-(d ** 2))
    w = w / (w.sum() + 1e-10)
    return w

from models.NN import ConditionalGeneralNet, ConditionalGeneralResNet, GeneralNet, GeneralNetBunny, GeneralNetPosEnc, GeneralResNet
from models.lip_ffn import LipschitzConditionalFFN
from models.lip_mlp import CondLipMLP
from models.lip_siren import CondLipSIREN
from models.net_w_partials import NetWithPartials
from models.siren import ConditionalSIREN, LatentModulatedSiren
from models.wire import ConditionalWIRE
from models.structured_lego import StructuredLegoModel
from functools import cmp_to_key


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_is_out_mask(x, bounds):
    out_mask = (x < bounds[:, 0]).any(1) | (x > bounds[:, 1]).any(1)
    return out_mask

def is_every_n_epochs_fulfilled(epoch, config, key):
    if key not in config:
        return False
    is_fulfilled = (epoch % config[key] == 0) or (epoch == config['max_epochs'] - 1)
    return is_fulfilled

def do_plot(config, epoch, key=None):
        """
        Checks if the plot specified by the key should be produced.
        First checks if the output is set: if not, no need to produce the plot.
        Then, checks if the key is set and true.
        If the key is not specified, only checks the global plotting behaviour.
        """
        is_output_set = config['plot']['fig_wandb']

        # if no output is set, no need to produce the plot
        if is_output_set == False:
            return False
        
        # if no key is specified, only check the global plotting behaviour
        if key is None:
            return is_output_set
        
        # otherwise, check if the key is set
        if key not in config:
            return False
        
        val = config[key]
        if isinstance(val, bool):
            return val
        
        if isinstance(val, dict):
            # if the val is a tuple, it is (do_plot, plot_interval)
            if val['on'] == False:
                return False
            else:
                assert val['interval'] % config['plot_every_n_epochs'] == 0, f'plot_interval must be a multiple of plot_every_n_epochs'
                return (epoch % val['interval'] == 0) or (epoch == config['max_epochs'])
        
        raise ValueError(f'Unknown value for key {key}: {val}')

def t_(x):
    return torch.tensor(x, dtype=torch.float32)


def _safe_int(x, default=0):
    """Convert to int; use default if value is unresolved (e.g. '${vars.nz}')."""
    if isinstance(x, int):
        return x
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def resolve_problem_config(config):
    """
    Return (prob_cfg, problem_sampling_kwargs) with ${vars.xxx} replaced by LEGO defaults.
    Use when config was loaded from YAML without variable resolution (e.g. notebook).
    """
    prob_cfg = dict(config.get('problem', {'problem_str': 'lego_1xN', 'n_studs': 4}))
    if isinstance(prob_cfg.get('problem_str'), str) and prob_cfg['problem_str'].startswith('${'):
        prob_cfg['problem_str'] = 'lego_1xN'
    sampling = dict(config.get('problem_sampling', {'nx': 3}))
    defaults = {
        'nx': 3, 'n_points_surface': 20000, 'n_points_domain': 2048,
        'n_points_envelope': 8192, 'n_points_interfaces': 4096, 'n_points_normals': 4096,
    }
    for k in list(sampling.keys()):
        v = sampling[k]
        if isinstance(v, str) and '${' in v:
            sampling[k] = defaults.get(k, 3)
        elif isinstance(v, str):
            sampling[k] = _safe_int(v, defaults.get(k, 3))
    return prob_cfg, sampling


def get_problem(problem_config, **kwargs):
    # Resolve unresolved YAML vars (e.g. ${vars.problem_str}) when config is used without resolve_problem_config
    problem_config = dict(problem_config)
    if isinstance(problem_config.get('problem_str'), str) and problem_config['problem_str'].startswith('${'):
        problem_config['problem_str'] = 'lego_1xN'
    if problem_config['problem_str'] == 'obstacle':
        from GINN.problems.problem_obstacle import ProblemObstacle
        return ProblemObstacle(**problem_config, **kwargs)
    elif problem_config['problem_str'] == 'wheel':
        from GINN.problems.problem_wheel import ProblemWheel
        return ProblemWheel(**problem_config, **kwargs)
    elif problem_config['problem_str'] == 'pipes':
        from GINN.problems.problem_pipes import ProblemPipes
        return ProblemPipes(**problem_config, **kwargs)
    elif problem_config['problem_str'] == 'mbb_beam_2d':
        from GINN.problems.problem_mbb_beam_2d import ProblemMBBBeam2d
        return ProblemMBBBeam2d(**problem_config, **kwargs)
    elif problem_config['problem_str'] == 'cantilever_2d':
        from GINN.problems.problem_cantilever_2d import ProblemCantilever2d
        return ProblemCantilever2d(**problem_config, **kwargs)
    elif problem_config['problem_str'] == 'simjeb':
        from GINN.problems.problem_simjeb import ProblemSimjeb
        return ProblemSimjeb(**problem_config, **kwargs)
    elif problem_config['problem_str'] == 'lego_1xN':
        from GINN.problems.problem_lego_1xN import ProblemLego1xN
        return ProblemLego1xN(**problem_config, **kwargs)
    else:
        raise ValueError(f'Unknown problem type: {problem_config["problem_str"]}')


def get_activation(act_str):
    if act_str == 'relu':
        activation = torch.relu
    elif act_str == 'softplus':
        activation = torch.nn.Softplus(beta=10)
    elif act_str == 'celu':
        activation = torch.celu
    elif act_str == 'sin':
        activation = torch.sin
    elif act_str == 'tanh':
        activation = torch.tanh
    else:
        activation = None
        # print(f'activation not set')

    return activation



def get_model(model_str, nx, nz, layers, ny=1, activation=None, w0=None, w0_initial=None, 
              wire_scale=None, n_ffeat=None, ffn_sigma=None, **kwargs):
    
    act = get_activation(activation)
    # cond_wire extras (LEGO): tiled coords (+2), dist_to_edge_x (+1)
    use_tiled_coords = kwargs.get('use_tiled_coords', False)
    use_dist_to_edge = kwargs.get('use_dist_to_edge', False)
    n_studs_vals = kwargs.get('n_studs_values') or []
    use_dist_to_edge = use_dist_to_edge and (model_str == 'cond_wire') and (n_studs_vals is not None and len(n_studs_vals) > 0)
    input_size = nx + nz
    if model_str == 'cond_wire':
        if use_tiled_coords:
            input_size += 2
        if use_dist_to_edge:
            input_size += 1
    layers.insert(0, input_size)
    layers.append(ny)  ## output layer

    if model_str == 'siren':
        raise ValueError('SIREN is not supported yet; the layers and in_features are not well defined')
        # model = SIREN(layers=layers, in_features=config['nx'], out_features=1, w0=config['w0'], w0_initial=config['w0_initial'])
    elif model_str == 'cond_siren':
        model = ConditionalSIREN(layers=layers, w0=w0, w0_initial=w0_initial, **kwargs)
    elif model_str == 'lip_siren':
        model = CondLipSIREN(layers=layers, nz=nz, w0=w0, w0_initial=w0_initial)
    elif model_str == 'lip_mlp':
        model = CondLipMLP(layers=layers)
    elif model_str == 'comod_siren':
        model = LatentModulatedSiren(layers=layers, w0=w0, w0_initial=w0_initial, latent_dim=nz)
    elif model_str == 'cond_wire':
        wire_kw = dict(kwargs)
        if 'stud_spacing_normalized' in wire_kw:
            wire_kw['stud_spacing'] = wire_kw.pop('stud_spacing_normalized', 1.0)
        if 'stud_spacing_y_normalized' in wire_kw:
            wire_kw['stud_spacing_y'] = wire_kw.pop('stud_spacing_y_normalized')
        model = ConditionalWIRE(layers=layers, first_omega_0=w0_initial, hidden_omega_0=w0, scale=wire_scale, **wire_kw)
    elif model_str == 'grid_mock':
        model = ConditionalGridMock(**kwargs)
    elif model_str == 'general_net':
        model = GeneralNet(ks=layers, act=act)
    elif model_str == 'general_resnet':
        model = GeneralResNet(ks=layers, act=act)
    elif model_str == 'cond_general_net':
        model = ConditionalGeneralNet(ks=layers, act=act)
    elif model_str == 'cond_general_resnet':
        model = ConditionalGeneralResNet(ks=layers, act=act)
    elif model_str == 'lip_ffn':
        model = LipschitzConditionalFFN(layers=layers, nz=nz, n_ffeat=n_ffeat, sigma=ffn_sigma)
    elif model_str == 'bunny':
        model = GeneralNetBunny(act='sin')
    elif model_str == 'structured_lego':
        # Structured LEGO decoder: predicts geometric params, computes analytic SDF
        model = StructuredLegoModel(
            condition_dim=kwargs.get('condition_dim', nz),
            z_dim=kwargs.get('structured_z_dim', 0),
            hidden_dims=kwargs.get('structured_hidden', [64, 64]),
            max_studs=kwargs.get('max_studs', 8),
            predict_stud_params=kwargs.get('predict_stud_params', True),
            smooth_k=kwargs.get('smooth_k', 0.02),
            use_studs=kwargs.get('use_studs', True)
        )
    else:
        raise ValueError(f'model not specified properly in config: {model}')
    return model



def geometric_mean(input_x, dim=0):
    '''
    Compute the geometric mean of the input tensor along the specified dimension.
    Is numerically stable for large values. For small values, taking the log first might be worse.
    '''
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

def inverse_geometric_mean(input_x, dim=0):
    '''
    Compute the inverse geometric mean of the input tensor along the specified dimension.
    Is numerically stable for large values. For small values, taking the log first might be worse.
    '''
    log_x = torch.log(input_x)
    return torch.exp(-torch.mean(log_x, dim=dim))

def compute_grad_norm(parameters):
    """
    Computes the norm of the gradients of the parameters.
    Args:
        parameters (iterable): An iterable of torch.Tensor containing the parameters of the model.
    """
    params_list = list(parameters)
    grads = [param.grad.detach().flatten() for param in params_list if param.grad is not None]
    if len(grads) == 0:
        device = params_list[0].device if params_list else None
        dtype = params_list[0].dtype if params_list else torch.get_default_dtype()
        return torch.tensor(0.0, device=device, dtype=dtype)
    grad_norm = torch.cat(grads).norm()
    return grad_norm

def fuzzy_lexsort(keys, tol):
    """
    Perform a lexsort on the keys with a tolerance for equality.

    Parameters:
    keys (list of arrays): The keys to sort by.
    tol (float): The tolerance within which two numbers are considered equal.

    Returns:
    sorted_indices (array): The indices that would sort the keys lexicographically with the given tolerance.
    """
    # Ensure keys are numpy arrays
    keys = [np.asarray(key) for key in keys]

    # Create a custom comparison function
    def fuzzy_compare(a, b):
        if abs(a[1] - b[1]) <= tol:
            return 0
        elif a[1] < b[1]:
            return -1
        else:
            return 1

    # Create a sorted index array for each key
    sorted_indices = np.arange(len(keys[0]))
    for k in reversed(keys):
        sorted_indices_delta = [i for i, x in sorted(enumerate(k[sorted_indices]), key=cmp_to_key(fuzzy_compare))]
        # sorted_indices_delta = sorted(k, key=cmp_to_key(fuzzy_compare))
        sorted_indices = sorted_indices[sorted_indices_delta]
    return np.array(sorted_indices)

def combine_dicts(dicts):
    """
    Combines a list of dictionaries into a single dictionary.
    Args:
        dicts (list): A list of dictionaries.
    Returns:
        combined_dict (dict): The combined dictionary.
    """
    combined_dict = {}
    for d in dicts:
        combined_dict.update(d)
    return combined_dict

def idx_of_tensor_in_list(target_tensor, tensor_list):
    for i, tensor in enumerate(tensor_list):
        if torch.equal(tensor, target_tensor):
            return i
    return -1