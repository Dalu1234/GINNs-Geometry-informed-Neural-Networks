import math
import random
import time
import torch
from tqdm import trange
import logging
from functools import partial

import wandb
from GINN.data.simjeb_dataloader import SimJebDataloader
from GINN.numerical_boundary_helper import NumericalBoundaryHelper
from GINN.shape_boundary_helper import ShapeBoundaryHelper
from GINN.data.simjeb_dataset import SimJebDataset
from GINN.evaluation.jeb_meter import JebMeter
from GINN.evaluation.simple_obst_meter import SimpleObstacleMeter
from models.net_w_partials import NetWithPartials
from train.opt.opt_util import get_opt, opt_step
from train.train_utils.loss_calculator_alm import AdaptiveAugmentedLagrangianLoss
from train.train_utils.loss_calculator_manual import ManualWeightedLoss
from train.train_utils.loss_keys import LossKey, get_loss_keys_and_lambdas
from train.train_utils.autoclip import AutoClip

from GINN.plot.plotter_2d import Plotter2d
from GINN.plot.plotter_3d import Plotter3d
from GINN.ph.ph_plotter import PHPlotter
from GINN.ph.ph_manager import PHManager
from GINN.speed.timer import Timer
from train.train_utils.latent_sampler import LatentSampler
from train.train_utils.violation_buffer import ViolationBuffer
from util.checkpointing import load_model_optim_sched, load_conditional_prior, save_model_every_n_epochs
from util.misc import combine_dicts, do_plot, get_model, get_problem, is_every_n_epochs_fulfilled
from train.losses_ginn import *
from train.losses_diversity import diversity_loss_chamfer, diversity_loss_contrastive, diversity_loss_volume_symmetric_difference, diversity_loss_combined
from train.losses_connectivity_v2 import CachedConnectivityLossV2 as CachedConnectivityLoss
from train.train_utils.loss_calculator_alm import AdaptiveAugmentedLagrangianLoss


class Trainer():
    
    def __init__(self, config, mp_manager, mp_pool_scc, mp_top) -> None:
        self.config = config
        self.mpm = mp_manager
        self.logger = logging.getLogger('trainer')
        # LEGO 1xN / NxN: condition on N (studs), N_y (1xN vs NxM), and/or height -> model gets nz + extra dims
        self.condition_on_n_studs = (
            config.get('problem', {}).get('problem_str') == 'lego_1xN'
            and config.get('condition_on_n_studs', False)
        )
        self.condition_on_n_studs_y = (
            config.get('problem', {}).get('problem_str') == 'lego_1xN'
            and config.get('condition_on_n_studs_y', False)
        )
        self.condition_on_height = (
            config.get('problem', {}).get('problem_str') == 'lego_1xN'
            and config.get('condition_on_height', False)
        )
        model_kw = dict(config['model'])
        model_kw['layers'] = list(model_kw.get('layers', [128, 128, 128]))  # copy; get_model mutates
        if self.condition_on_n_studs:

            # Prefer top-level; LEGO config may keep it under vars after merge
            self.n_studs_values = list(
                config.get('n_studs_values')
                or config.get('vars', {}).get('n_studs_values')
                or [2, 3]
            )
            self.logger.info(f'n_studs_values (brick types): {self.n_studs_values}')
            model_kw['nz'] = model_kw['nz'] + 1  # shape latent + 1 for N
        if self.condition_on_n_studs_y:
            self.n_studs_y_values = list(config.get('n_studs_y_values', [1, 2, 4]))  # 1 = 1xN, 2+ = NxM
            model_kw['nz'] = model_kw['nz'] + 1  # +1 for N_y
        if self.condition_on_height:
            self.height_values = list(config.get('height_values', [0.75, 1.0, 1.25]))
            model_kw['nz'] = model_kw['nz'] + 1  # +1 for height scale
        self.model = get_model(**model_kw)
        self.netp = NetWithPartials.create_from_model(self.model, **model_kw)
        if self.condition_on_n_studs or self.condition_on_n_studs_y or self.condition_on_height:
            # One problem per (N, N_y, height) or subset; constraints unchanged per problem
            # Build problems from n_studs_values etc. only (do not create default problem with config['problem'].n_studs: 4)
            self.problems = {}
            if self.condition_on_n_studs:
                self._n_list = self.n_studs_values
            else:
                self._n_list = [self.config['problem'].get('n_studs', 4)]
            if self.condition_on_n_studs_y:
                self._ny_list = self.n_studs_y_values  # 1 = 1xN (n_studs_y=None), 2+ = NxM
            else:
                self._ny_list = [1]
            if self.condition_on_height:
                self._h_list = self.height_values
            else:
                self._h_list = [1.0]
            for n in self._n_list:
                for ny in self._ny_list:
                    for h in self._h_list:
                        prob_cfg = {**self.config['problem'], 'n_studs': n, 'height_scale': h}
                        if self.condition_on_n_studs_y:
                            prob_cfg['n_studs_y'] = None if ny == 1 else ny  # 1 = 1xN
                        if self.condition_on_height and self.condition_on_n_studs_y:
                            key = (n, ny, h)
                        elif self.condition_on_height:
                            key = (n, h)
                        elif self.condition_on_n_studs_y:
                            key = (n, ny)
                        else:
                            key = n
                        self.problems[key] = get_problem(problem_config=prob_cfg, **self.config['problem_sampling'])
            n0, ny0, h0 = self._n_list[0], self._ny_list[0], self._h_list[0]
            key0 = (n0, ny0, h0) if self.condition_on_height and self.condition_on_n_studs_y else (n0, ny0) if self.condition_on_n_studs_y else (n0, h0) if self.condition_on_height else n0
            self.problem = self.problems[key0]
        else:
            self.problem = get_problem(problem_config=self.config['problem'], **self.config['problem_sampling'])

        self.timer = Timer(**self.config['timer'], lock=mp_manager.get_lock())
        self.mpm.set_timer(self.timer)  ## weak circular reference
        if self.config['nx'] == 2:
            self.plotter = Plotter2d(envelope=self.problem.envelope, bounds=self.problem.bounds, **self.config['plot'])
        elif self.config['nx'] == 3:
            self.plotter = Plotter3d(bounds=self.problem.bounds, **self.config['plot'])
        
        self.ph_manager = None
        if self.config['lambda_scc'] > 0:
            self.ph_manager = PHManager(bounds=self.problem.bounds, netp=self.netp, func_inside_envelope=self.problem.is_inside_envelope,
                                    mp_pool=mp_pool_scc, **self.config['ph'])
        self.ph_plotter = PHPlotter(iso_level=self.config['ph']['iso_level'], fig_size=self.config['plot']['fig_size'])
        if self.config.get('do_numerical_surface_points', True):
            self.shape_boundary_helper = NumericalBoundaryHelper(netp=self.netp, mp_manager=self.mpm, plotter=self.plotter, 
                                                         x_interface=self.problem.sample_from_interface()[0], bounds=self.problem.bounds, **self.config['surface'])
        else:
            self.shape_boundary_helper = ShapeBoundaryHelper(netp=self.netp, mp_manager=self.mpm, plotter=self.plotter, 
                                                         x_interface=self.problem.sample_from_interface()[0], bounds=self.problem.bounds,
                                                         **self.config['surface'])
        self.auto_clip = AutoClip(**config['grad_clipping'])
        self.z_sampler = LatentSampler(**config['latent_sampling'])
        
        # FEM interface
        self.feni = None
        if self.config['lambda_comp'] > 0 or self.config['lambda_vol'] > 0:
            from GINN.fenitop.fenitop_mp import FenitopMP
            from GINN.fenitop.heaviside_filter import heaviside
            feni_kwargs = self.config['FEM'].copy()
            
            if self.config['problem']['problem_str'] == 'simjeb':
                feni_kwargs['interface_pts'] = self.problem.constr_pts_dict['interface']
                feni_kwargs['envelope_mesh'] = self.problem.mesh_env
                feni_kwargs['envelope_unnormalized'] = self.problem.mesh_env.bounds.T
                feni_kwargs['bounds_unnormalized'] = self.problem.bounds
            else:
                feni_kwargs['envelope_unnormalized'] = self.problem.envelope_unnormalized.cpu().numpy()
                feni_kwargs['bounds_unnormalized'] = self.problem.bounds_unnormalized
            self.feni = FenitopMP(device=torch.get_default_device(), bz=self.config['ginn_bsize'], mp_pool=mp_top, np_func_inside_envelope=self.problem.is_inside_envelope, **feni_kwargs)
            self.beta = self.config['beta_sched']['init']
            
        # data
        if ('lambda_data' in config) and (config['lambda_data'] > 0):
            dataset = SimJebDataset(**config['data'], seed=config['seed'])
            self.dataloader = SimJebDataloader(dataset, config['n_points_data'])
        
        # metrics
        self.meter = None
        if config['problem'] in ['obstacle', 'double_obstacle']:
            self.meter = SimpleObstacleMeter.create_from_problem_sampler(self.problem, **config['metrics'])
        elif config['problem'] == 'simjeb':
            self.meter = JebMeter(config)
            # self.config['n_points_envelope'] = self.config['n_points_envelope'] // 3 * 3 # what's this for? This is redundant. Where parentheses misplaced?

        self.p_surface = None
        self.weights_surf_pts = None
        self.cur_plot_epoch = 0
        self.log_history_dict = {}
        
        # Connectivity loss (Morse theory V2) for generative training
        # V2 improvements: near-surface filtering, clamped Newton steps
        self.cached_connectivity_loss = None
        if self.config.get('lambda_connectivity', 0) > 0:
            self.cached_connectivity_loss = CachedConnectivityLoss(
                bounds=self.problem.bounds,
                n_samples=self.config.get('connectivity_n_samples', 1000),
                n_iter=self.config.get('connectivity_n_iter', 30),
                grad_tol=self.config.get('connectivity_grad_tol', 1e-2),
                update_every_n_epochs=self.config.get('connectivity_update_every_n_epochs', 50),
                near_surface_threshold=self.config.get('connectivity_near_surface_threshold', 0.15),
                max_step_size=self.config.get('connectivity_max_step_size', 0.3),
            )

        # Violation buffer: sample (z, c) proportional to interface+envelope violation
        vb_cfg = self.config.get('violation_buffer', {}) or {}
        self.use_violation_buffer = (
            vb_cfg.get('use', False)
            and (self.condition_on_n_studs or self.condition_on_n_studs_y or self.condition_on_height)
        )
        self.violation_buffer = None
        if self.use_violation_buffer:
            nz_base = self.config['latent_sampling']['nz']
            c_keys = list(self.problems.keys())
            z_interval = self.config['latent_sampling'].get('z_sample_interval', [0.0, 0.1])
            initial_size_per_c = vb_cfg.get('initial_size_per_c', 50)
            self.violation_buffer = ViolationBuffer(
                nz_base=nz_base,
                c_keys=c_keys,
                z_interval=z_interval,
                initial_size_per_c=initial_size_per_c,
                device=torch.get_default_device(),
                eps=vb_cfg.get('eps', 1e-8),
            )
            self.logger.info(f'Violation buffer: {len(self.violation_buffer)} entries ({initial_size_per_c} per c_key)')

        # Conditional latent prior p(z_base | c): learned per-condition distribution over shape latent
        self.conditional_prior = None
        self.nz_base_prior = None
        self.c_dim_prior = None
        if self.config.get('lambda_prior', 0) > 0 and (self.condition_on_n_studs or self.condition_on_n_studs_y or self.condition_on_height):
            from train.train_utils.conditional_prior import ConditionalPrior
            self.nz_base_prior = self.config['latent_sampling']['nz']
            self.c_dim_prior = (1 if self.condition_on_n_studs else 0) + (1 if self.condition_on_n_studs_y else 0) + (1 if self.condition_on_height else 0)
            prior_hidden = self.config.get('prior_hidden', [32, 32])
            self.conditional_prior = ConditionalPrior(
                c_dim=self.c_dim_prior,
                z_base_dim=self.nz_base_prior,
                hidden_dims=prior_hidden,
                min_sigma=self.config.get('prior_min_sigma', 1e-3),
            )
            self.logger.info(f'Conditional prior: z_base_dim={self.nz_base_prior}, c_dim={self.c_dim_prior}')
        
        # loss balancing
        self.scalar_loss_keys, self.field_loss_keys, lambda_dict, objective_key = get_loss_keys_and_lambdas(self.config)
        self.all_loss_keys = self.scalar_loss_keys + self.field_loss_keys
        self.loss_dispatcher = self.init_loss_dispatcher()
        if self.config['use_augmented_lagrangian']:
            self.loss_calculator = AdaptiveAugmentedLagrangianLoss(scalar_loss_keys=self.scalar_loss_keys, obj_key=objective_key, 
                                                                    lambda_dict=lambda_dict, field_loss_keys=self.field_loss_keys, **self.config['adaptive_penalty'])
        else:
            self.loss_calculator = ManualWeightedLoss(scalar_loss_keys=self.scalar_loss_keys, lambda_dict=lambda_dict, field_loss_keys=self.field_loss_keys)
        
        self.p = None

    def _sync_problem_dependents(self):
        """After switching self.problem (e.g. LEGO N), update helpers that cache bounds/interface."""
        if self.config['nx'] == 3:
            self.plotter.bounds = self.problem.bounds.cpu().numpy()
        if self.shape_boundary_helper is not None and hasattr(self.shape_boundary_helper, 'set_problem'):
            self.shape_boundary_helper.set_problem(self.problem)
        if self.ph_manager is not None and hasattr(self.ph_manager, 'set_problem'):
            self.ph_manager.set_problem(self.problem)
        if self.cached_connectivity_loss is not None:
            self.cached_connectivity_loss.bounds = self.problem.bounds

    def _unpack_ckey(self, c_key):
        """Return (N, ny, h) from c_key (tuple or int) for building conditioning cols."""
        if isinstance(c_key, tuple):
            if len(c_key) == 3:
                return c_key[0], c_key[1], c_key[2]
            if len(c_key) == 2:
                return c_key[0], c_key[1], 1.0
            return c_key[0], 1, 1.0
        return c_key, 1, 1.0

    def _compute_violation_for_z_c(self, z_single, c_key):
        """Interface + envelope loss (no_grad) for one (z, c) pair."""
        problem = self.problems[c_key]
        with torch.no_grad():
            l_if = loss_if(z=z_single, netp=self.netp, p_sampler=problem, level_set=self.config['level_set'])
            l_env = loss_env(z=z_single, netp=self.netp, p_sampler=problem, level_set=self.config['level_set'], nf_is_density=self.config['nf_is_density'])
        return (l_if + l_env).item()
        
    def train(self):
        
        # get optimizer and scheduler (include conditional prior params when used)
        params = list(self.model.parameters())
        if getattr(self, 'conditional_prior', None) is not None:
            params += list(self.conditional_prior.parameters())
        opt = get_opt(self.config['opt'], self.config, params)
        sched = None
        if self.config.get('use_scheduler', False):
            def warm_and_decay_lr_scheduler(step: int):
                return self.config['scheduler_gamma'] ** (step / self.config['decay_steps'])
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warm_and_decay_lr_scheduler)
        # maybe load model, optimizer and scheduler
        self.model, opt, sched = load_model_optim_sched(self.config, self.model, opt, sched)
        if getattr(self, 'conditional_prior', None) is not None:
            load_conditional_prior(self.config, self.conditional_prior, device=torch.get_default_device())
        
        # get z
        z = self.z_sampler.train_z()
        # In generative mode, don't use corner anchors - pure random sampling
        if self.config.get('training_mode', 'single') == 'generative':
            z_corners = torch.zeros(0, z.shape[1])  # empty tensor with correct nz
            self.logger.info('Generative mode: z will be resampled every batch')
        else:
            z_corners = self.z_sampler.get_z_corners(len(self.config['data'].get('simjeb_ids', [])))
        z_val = self.z_sampler.val_z()
        if self.condition_on_n_studs or self.condition_on_n_studs_y or self.condition_on_height:
            cols = []
            if self.condition_on_n_studs:
                n_min, n_max = min(self.n_studs_values), max(self.n_studs_values)
                n_col = torch.tensor(
                    [(self.n_studs_values[i % len(self.n_studs_values)] - n_min) / max(n_max - n_min, 1) for i in range(z_val.shape[0])],
                    device=z_val.device, dtype=z_val.dtype
                ).unsqueeze(1)
                cols.append(n_col)
            if self.condition_on_n_studs_y:
                ny_min, ny_max = min(self.n_studs_y_values), max(self.n_studs_y_values)
                ny_col = torch.tensor(
                    [(self.n_studs_y_values[i % len(self.n_studs_y_values)] - ny_min) / max(ny_max - ny_min, 1) for i in range(z_val.shape[0])],
                    device=z_val.device, dtype=z_val.dtype
                ).unsqueeze(1)
                cols.append(ny_col)
            if self.condition_on_height:
                h_min, h_max = min(self.height_values), max(self.height_values)
                h_col = torch.tensor(
                    [(self.height_values[i % len(self.height_values)] - h_min) / max(h_max - h_min, 1e-6) for i in range(z_val.shape[0])],
                    device=z_val.device, dtype=z_val.dtype
                ).unsqueeze(1)
                cols.append(h_col)
            z_val = torch.cat([z_val] + cols, dim=1)
        self.logger.info(f'Initial z (resampling: {self.config["latent_sampling"]["z_sample_method"]}): {z}')        
        self.logger.info(f'z_corners (not resampled): {z_corners}')
        self.logger.info(f'z_val (not resampled): {z_val}')
        n_train_shapes_to_plot = min(len(z) + len(z_corners), self.config['train_plot_max_n_shapes'])
        
        # training/validation loop
        for epoch in (pbar := trange(self.config['max_epochs'], leave=True, position=0, colour="yellow")):
            cur_log_dict = {}
            self.mpm.update_epoch(epoch)
            batch = self.dataloader.get_data_batch() if self.config['lambda_data'] > 0 else None
            opt.zero_grad()

            ## validation
            if epoch > 0 and is_every_n_epochs_fulfilled(epoch, self.config, 'valid_every_n_epochs'):
                if self.condition_on_n_studs or self.condition_on_n_studs_y or self.condition_on_height:
                    n0, ny0, h0 = self._n_list[0], self._ny_list[0], self._h_list[0]
                    key0 = (n0, ny0, h0) if self.condition_on_height and self.condition_on_n_studs_y else (n0, ny0) if self.condition_on_n_studs_y else (n0, h0) if self.condition_on_height else n0
                    self.problem = self.problems[key0]
                    self._sync_problem_dependents()
                    z_val_single = z_val[:1]
                else:
                    z_val_single = z_val
                self.model.eval()
                # plot validation shapes with variety: one mesh per brick type (each row of z_val)
                if self.config['nx'] == 2:
                    Y = self.problem.recalc_output(self.netp, z_val_single, **self.config['meshing'])
                    if self.config['lambda_comp'] > 0:
                        Y = heaviside(Y, beta=self.beta, nf_is_density=self.config['nf_is_density'])
                    self.plotter.reset_output(Y.cpu().numpy(), epoch=epoch)
                    self.mpm.plot(self.plotter.plot_shape, 'val_plot_shape', kwargs_dict={'fig_label': 'Val Boundary', 'is_validation': True})
                elif self.config['nx'] == 3:
                    if self.condition_on_n_studs or self.condition_on_n_studs_y or self.condition_on_height:
                        n_rows, n_cols = self.plotter.val_plot_grid
                        expected = n_rows * n_cols
                        val_verts_faces = []
                        for i in range(z_val.shape[0]):
                            n_i = self.n_studs_values[i % len(self.n_studs_values)]
                            ny_i = self.n_studs_y_values[i % len(self.n_studs_y_values)]
                            h_i = self.height_values[i % len(self.height_values)]
                            if self.condition_on_height and self.condition_on_n_studs_y:
                                key_i = (n_i, ny_i, h_i)
                            elif self.condition_on_n_studs_y:
                                key_i = (n_i, ny_i)
                            elif self.condition_on_height:
                                key_i = (n_i, h_i)
                            else:
                                key_i = n_i
                            if key_i not in self.problems:
                                continue
                            self.problem = self.problems[key_i]
                            self._sync_problem_dependents()
                            z_row = z_val[i : i + 1]
                            out_i = self.problem.recalc_output(self.netp, z_row, **self.config['meshing'])
                            if out_i is not None and len(out_i) > 0:
                                val_verts_faces.append(out_i[0])
                        if len(val_verts_faces) == expected:
                            self.plotter.reset_output(val_verts_faces, epoch=epoch)
                            self.mpm.plot(self.plotter.plot_shape, 'val_plot_shape', kwargs_dict={'fig_label': 'Val Boundary', 'is_validation': True})
                        else:
                            self.plotter.reset_output(self.problem.recalc_output(self.netp, z_val_single, **self.config['meshing']), epoch=epoch)
                            self.mpm.plot(self.plotter.plot_shape, 'val_plot_shape', kwargs_dict={'fig_label': 'Val Boundary', 'is_validation': True})
                        self.problem = self.problems[key0]
                        self._sync_problem_dependents()
                    else:
                        self.plotter.reset_output(self.problem.recalc_output(self.netp, z_val_single, **self.config['meshing']), epoch=epoch)
                        self.mpm.plot(self.plotter.plot_shape, 'val_plot_shape', kwargs_dict={'fig_label': 'Val Boundary', 'is_validation': True})
                else:
                    self.mpm.plot(self.plotter.plot_shape, 'val_plot_shape', kwargs_dict={'fig_label': 'Val Boundary', 'is_validation': True})
                
                # compute validation metrics over multiple brick types (each row of z_val = one brick type)
                if epoch > 0 and is_every_n_epochs_fulfilled(epoch, self.config, 'val_shape_metrics_every_n_epochs'):
                    if self.condition_on_n_studs or self.condition_on_n_studs_y or self.condition_on_height:
                        val_meshes = []
                        for i in range(z_val.shape[0]):
                            # key for row i: same indexing as z_val cols (n, ny, h)
                            n_i = self.n_studs_values[i % len(self.n_studs_values)]
                            ny_i = self.n_studs_y_values[i % len(self.n_studs_y_values)]
                            h_i = self.height_values[i % len(self.height_values)]
                            if self.condition_on_height and self.condition_on_n_studs_y:
                                key_i = (n_i, ny_i, h_i)
                            elif self.condition_on_n_studs_y:
                                key_i = (n_i, ny_i)
                            elif self.condition_on_height:
                                key_i = (n_i, h_i)
                            else:
                                key_i = n_i
                            if key_i not in self.problems:
                                continue
                            self.problem = self.problems[key_i]
                            self._sync_problem_dependents()
                            z_row = z_val[i : i + 1]
                            mesh_or_contour_i = self.problem.get_mesh_or_contour(self.netp.f_, self.netp.params, z_row)
                            if mesh_or_contour_i is not None and len(mesh_or_contour_i) > 0:
                                val_meshes.append(mesh_or_contour_i[0])
                        if val_meshes:
                            self.mpm.metrics(self.meter.get_average_metrics_as_dict, arg_list=[val_meshes], kwargs_dict={'prefix': 'm_val_'})
                    else:
                        mesh_or_contour = self.problem.get_mesh_or_contour(self.netp.f_, self.netp.params, z_val_single)
                        if mesh_or_contour is not None:
                            self.mpm.metrics(self.meter.get_average_metrics_as_dict, arg_list=[mesh_or_contour], kwargs_dict={'prefix': 'm_val_'})

            ## training
            self.model.train()
            # Violation buffer: sample (z, c) proportional to interface+envelope violation
            if self.use_violation_buffer:
                vb_cfg = self.config.get('violation_buffer', {}) or {}
                c_keys_list = list(self.problems.keys())
                block_epochs = vb_cfg.get('block_epochs_per_type', 0) or self.config.get('block_epochs_per_type', 0) or 0
                if block_epochs > 0:
                    type_index = (epoch // block_epochs) % len(c_keys_list)
                    c_key = c_keys_list[type_index]
                else:
                    c_key = random.choice(c_keys_list)
                z_bases, indices = self.violation_buffer.sample_batch(
                    c_key, self.config['ginn_bsize'],
                    temperature=vb_cfg.get('sample_temperature', 1.0),
                    random_fraction=vb_cfg.get('random_fraction', 0.1),
                )
                N, ny, h = self._unpack_ckey(c_key)
                self.problem = self.problems[c_key]
                self._sync_problem_dependents()
                cols = []
                if self.condition_on_n_studs:
                    n_min, n_max = min(self.n_studs_values), max(self.n_studs_values)
                    n_norm = (N - n_min) / max(n_max - n_min, 1)
                    cols.append(torch.full((z_bases.shape[0], 1), n_norm, device=z_bases.device, dtype=z_bases.dtype))
                if self.condition_on_n_studs_y:
                    ny_min, ny_max = min(self.n_studs_y_values), max(self.n_studs_y_values)
                    ny_norm = (ny - ny_min) / max(ny_max - ny_min, 1)
                    cols.append(torch.full((z_bases.shape[0], 1), ny_norm, device=z_bases.device, dtype=z_bases.dtype))
                if self.condition_on_height:
                    h_min, h_max = min(self.height_values), max(self.height_values)
                    h_norm = (h - h_min) / max(h_max - h_min, 1e-6)
                    cols.append(torch.full((z_bases.shape[0], 1), h_norm, device=z_bases.device, dtype=z_bases.dtype))
                z = torch.cat([z_bases] + cols, dim=1)
                self._vb_batch_ckey = c_key
                self._vb_batch_indices = indices
            else:
                # In generative mode, resample z every batch for world model training
                if self.config.get('training_mode', 'single') == 'generative':
                    z = self.z_sampler.train_z()
                elif epoch > 0 and is_every_n_epochs_fulfilled(epoch, self.config, 'reset_zlatents_every_n_epochs'):
                    z = self.z_sampler.train_z()
                # LEGO 1xN / NxN: condition on N, N_y, and/or height; sample per batch, switch problem, append to z
                if self.condition_on_n_studs or self.condition_on_n_studs_y or self.condition_on_height:
                    block_epochs = self.config.get('violation_buffer', {}).get('block_epochs_per_type', 0) or self.config.get('block_epochs_per_type', 0) or 0
                    if block_epochs > 0:
                        c_keys_list = list(self.problems.keys())
                        type_index = (epoch // block_epochs) % len(c_keys_list)
                        key = c_keys_list[type_index]
                        N, ny, h = self._unpack_ckey(key)
                    else:
                        N = self.n_studs_values[torch.randint(len(self.n_studs_values), (1,)).item()] if self.condition_on_n_studs else self.config['problem'].get('n_studs', 4)
                        ny = self.n_studs_y_values[torch.randint(len(self.n_studs_y_values), (1,)).item()] if self.condition_on_n_studs_y else 1
                        h = self.height_values[torch.randint(len(self.height_values), (1,)).item()] if self.condition_on_height else 1.0
                        if self.condition_on_height and self.condition_on_n_studs_y:
                            key = (N, ny, h)
                        elif self.condition_on_height:
                            key = (N, h)
                        elif self.condition_on_n_studs_y:
                            key = (N, ny)
                        else:
                            key = N
                    self.problem = self.problems[key]
                    self._sync_problem_dependents()
                    cols = []
                    if self.condition_on_n_studs:
                        n_min, n_max = min(self.n_studs_values), max(self.n_studs_values)
                        n_norm = (N - n_min) / max(n_max - n_min, 1)
                        cols.append(torch.full((z.shape[0], 1), n_norm, device=z.device, dtype=z.dtype))
                    if self.condition_on_n_studs_y:
                        ny_min, ny_max = min(self.n_studs_y_values), max(self.n_studs_y_values)
                        ny_norm = (ny - ny_min) / max(ny_max - ny_min, 1)
                        cols.append(torch.full((z.shape[0], 1), ny_norm, device=z.device, dtype=z.dtype))
                    if self.condition_on_height:
                        h_min, h_max = min(self.height_values), max(self.height_values)
                        h_norm = (h - h_min) / max(h_max - h_min, 1e-6)
                        cols.append(torch.full((z.shape[0], 1), h_norm, device=z.device, dtype=z.dtype))
                    z = torch.cat([z] + cols, dim=1)
            # Optionally sample z_base from conditional prior p(z_base | c)
            if getattr(self, 'conditional_prior', None) is not None and self.config.get('sample_z_from_prior_prob', 0) > 0:
                if random.random() < self.config['sample_z_from_prior_prob']:
                    c = z[:, -self.c_dim_prior:]
                    z_base_new = self.conditional_prior.sample(c)
                    z = torch.cat([z_base_new, z[:, self.nz_base_prior:]], dim=1)
            # only plot the GINN points in the first epoch, as they are memory-intensive
            # if self.feni.constraint_to_pts_dict if epoch > 0 else combine_dicts([self.feni.constraint_to_pts_dict, self.problem.constr_pts_dict])
            vis_pts_dict = {} if epoch > 0 else self.problem.constr_pts_dict
            if self.feni is not None: 
                vis_pts_dict = combine_dicts([vis_pts_dict, self.feni.constraint_to_pts_dict])
                
            ## reset base shape if needed (skip epoch 0 if skip_plot_epoch_0 to avoid long recalc_output + PH block)
            skip_plot_this_epoch = (epoch == 0 and self.config.get('skip_plot_epoch_0', False))
            if not skip_plot_this_epoch and do_plot(self.config, epoch) and is_every_n_epochs_fulfilled(epoch, self.config, 'plot_every_n_epochs'):
                z_plot = self.z_sampler.combine_z_with_z_corners(z, z_corners, n_train_shapes_to_plot)
                if self.config['plot_shape']:
                    if self.config['nx'] == 2:
                        Y = self.problem.recalc_output(self.netp, z_plot, **self.config['meshing'])
                        self.plotter.reset_output(Y.cpu().numpy(), epoch=epoch)
                        self.mpm.plot(self.plotter.plot_shape, 'plot_shape', arg_list=['Train Boundary', vis_pts_dict])
                    elif self.config['nx'] == 3:
                        self.plotter.reset_output(self.problem.recalc_output(self.netp, z_plot, **self.config['meshing']), epoch=epoch)
                        self.mpm.plot(self.plotter.plot_shape, 'plot_shape', arg_list=['Train Boundary', vis_pts_dict])
                    
                if self.config['lambda_comp'] > 0:
                    with torch.no_grad():
                        chosen_rho_field = self.netp(*tensor_product_xz(self.feni.fem_xy_torch_device, z_plot)).view(len(z_plot), -1)
                    # for some strange reason k3d.volume expects the axes as (z, y, x)
                    if self.config['nx'] == 2:
                        chosen_rho_field = einops.rearrange(chosen_rho_field[:, self.feni.sort_idcs], 'bz (y x) -> bz x y', x=self.feni.chosen_resolution[1])
                    elif self.config['nx'] == 3:
                        chosen_rho_field = einops.rearrange(chosen_rho_field[:, self.feni.sort_idcs], 'bz (x y z) -> bz z y x', x=self.feni.chosen_resolution[0], y=self.feni.chosen_resolution[1])
                    if not self.config['FEM']['use_filters']:
                        chosen_rho_field = heaviside(chosen_rho_field, beta=self.beta, nf_is_density=self.config['nf_is_density'])
                    self.mpm.plot(self.plotter.plot_grad_field, '', kwargs_dict={'grad_fields': chosen_rho_field.detach().cpu().numpy(), 'constraint_pts_dict': vis_pts_dict, 'fig_label': f'Rho field'})
                    
                if self.ph_manager is not None:
                    # subsample to n_train_shapes_to_plot
                    PH = self.ph_manager.calc_ph(z[:n_train_shapes_to_plot])
                    PH = [ph.get() for ph in PH]
                    self.mpm.plot(self.ph_plotter.plot_ph_diagram, 'plot_ph_diagram', arg_list=[PH], kwargs_dict={})

            ## compute metrics _before_ taking the GD step
            if epoch > 0 and is_every_n_epochs_fulfilled(epoch, self.config, 'shape_metrics_every_n_epochs'):
                mesh_or_contour = self.problem.get_mesh_or_contour(self.netp.f_, self.netp.params, z)
                if mesh_or_contour is not None:
                    self.mpm.metrics(self.meter.get_average_metrics_as_dict, arg_list=[mesh_or_contour], kwargs_dict={'prefix': 'm_train_'})
                
                if self.config['lambda_data'] > 0:
                    mesh_or_contour = self.problem.get_mesh_or_contour(self.netp.f_, self.netp.params, z_corners)
                    if mesh_or_contour is not None:
                        self.mpm.metrics(self.meter.get_average_metrics_as_dict, arg_list=[mesh_or_contour], kwargs_dict={'prefix': 'm_corners_'})
            
            ## gradients and optimizer step
            with Timer.record('compute losses'):
                loss_log_dict = opt_step(opt=opt, epoch=epoch, model=self.model, loss_and_backward_fn=self.losses_and_backward,
                                         z=z, z_corners=z_corners, batch=batch, auto_clip=self.auto_clip)
            cur_log_dict.update(loss_log_dict)
            if self.config.get('use_scheduler', False):
                sched.step()

            # Refresh violation scores for the (z, c) pairs we just used (post-update model); random slots have no buffer index
            if self.use_violation_buffer:
                violations = [self._compute_violation_for_z_c(z[i:i + 1], self._vb_batch_ckey) for i in range(z.shape[0])]
                valid = [(i, v) for i, v in zip(self._vb_batch_indices, violations) if i is not None]
                if valid:
                    idx, vals = zip(*valid)
                    self.violation_buffer.update_violations(list(idx), list(vals))

            # Skip ALM update when we skipped the step so multipliers don't grow on no-op epochs
            if not getattr(self, '_last_step_skipped', False):
                self.loss_calculator.adaptive_update()  # is a no-op for ManualWeightedLoss

            ## Async Logging
            cur_log_dict.update({
                'neg_loss_div': (-1) * loss_log_dict['loss_div'] if 'loss_div' in loss_log_dict else 0.0,
                'grad_norm_post_clip': self.auto_clip.get_last_gradient_norm(),
                'grad_clip': self.auto_clip.get_clip_value(),
                'lr': self.config['lr'] if sched is None else sched.get_last_lr(),
                'epoch': epoch,
                })
            self.log_history_dict[epoch] = cur_log_dict
            self.log_to_wandb(epoch)
            save_model_every_n_epochs(
                self.model, opt, sched, self.config, epoch,
                conditional_prior=getattr(self, 'conditional_prior', None),
            )
            pbar.set_description(f"{epoch}:" + " ".join([f"{k.replace('loss_', '')}:{v.item():.1e}" for k, v in loss_log_dict.items() if "loss_" in k and "unweighted" not in k]))
        
        ## Final logging
        self.log_to_wandb(epoch, await_all=True)
        ## Finished
        self.timer.print_logbook()


    def losses_and_backward(self, z, epoch, batch=None, z_corners=None):

        loss_dict = {}
        field_dict = {}

        # compute surface points
        surf_for_curv = LossKey('curv') in self.all_loss_keys and epoch >= self.config.get('start_curv', 0)
        surf_for_div = LossKey('div') in self.all_loss_keys and epoch >= self.config.get('start_div', 0) and self.config['diversity_pts_source'] == 'surface'
        surf_for_chamfer_div = LossKey('chamfer_div') in self.all_loss_keys and epoch >= self.config.get('start_chamfer_div', 0)
        surf_needs_update = self.p_surface is None or is_every_n_epochs_fulfilled(epoch, self.config['surface'], 'recompute_every_n_epochs')
        if (surf_for_curv or surf_for_div or surf_for_chamfer_div) and surf_needs_update:
            with Timer.record('shape_boundary_helper.get_surface_pts'):
                self.p_surface, self.weights_surf_pts = self.shape_boundary_helper.get_surface_pts(z, 
                                                                plot=is_every_n_epochs_fulfilled(epoch, self.config, 'plot_surface_pts_every_n_epochs'),
                                                                plot_max_shapes=self.config['train_plot_max_n_shapes'],
                                                                interface_cutoff=self.config['exclude_surface_points_close_to_interface_cutoff'])
        
        ## compute chamfer div loss
        if self.config['lambda_chamfer_div'] > 0:   
            if self.p_surface is not None and epoch >= self.config.get('start_chamfer_div', 0):
                y_surface, p_surface_subsampled, batch_CD, pdCDdy = self.loss_dispatcher['chamfer_div'](z=z, p_surface=self.p_surface)
                loss_dict[LossKey('chamfer_div')] = (batch_CD, pdCDdy)
                field_dict[LossKey('chamfer_div')] = y_surface
                
                if not torch.equal(pdCDdy.data, torch.zeros_like(pdCDdy.data)) and is_every_n_epochs_fulfilled(epoch, self.config, 'plot_every_n_epochs'):
                    self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_chamfer_grads',
                        arg_list=[p_surface_subsampled.detach().cpu().numpy(), 'Chamfer penalty'], kwargs_dict=dict(point_attribute=pdCDdy.data.detach().cpu().numpy()))

            else:
                loss_dict[LossKey('chamfer_div')] = (torch.tensor(0.0), torch.tensor(0.0))            
                field_dict[LossKey('chamfer_div')] = torch.tensor(0.0)
                            
        ## compute TO sublosses
        x_fem = None
        if self.config['lambda_comp'] > 0:
            x_fem = self.feni.fem_xy_torch_device
            if self.config['FEM']['add_x_offset']:
                x_fem = x_fem + (torch.rand(self.feni.fem_xy_torch_device.shape) - 0.5) * self.feni.grid_dist
            rho_flat = self.netp(*tensor_product_xz(x_fem, z)).squeeze(1)
            
            # update beta
            if epoch > self.config['beta_sched'].get('start_iter', 0) and epoch % self.config['beta_sched']['interval'] == 0:
                self.beta = math.exp2(min(math.log2(self.config['beta_sched']['init']) + (epoch / self.config['beta_sched']['stop_iter']) * math.log2(self.config['beta_sched']['max'] / self.config['beta_sched']['init']), 
                                    math.log2(self.config['beta_sched']['max'])))  # exponential increase; log formulation avoids numerical issues
                print(f'beta: {self.beta}')
            
            if not self.config['FEM']['use_filters']:
                rho_flat = heaviside(rho_flat, beta=self.beta, nf_is_density=self.config['nf_is_density'])
            
            if epoch > self.config['p_sched'].get('start_iter', 0) and epoch % self.config['p_sched']['interval'] == 0:
                self.p = min(self.config['p_sched']['init'] + (epoch / self.config['p_sched']['stop_iter'] * (self.config['p_sched']['max'] - self.config['p_sched']['init'])), 
                                    self.config['p_sched']['max'])  # exponential increase; log formulation avoids numerical issues
                print(f'p: {self.p}')
                rho_flat = rho_flat ** self.p
            
            rho_batch = einops.rearrange(rho_flat, '(bz n) -> bz n', bz=z.shape[0])
            (batch_C, batch_dCdrho), (batch_V, batch_dVdrho) = self.feni.calc_2d_TO_grad_field_batch(rho_batch=rho_batch, beta=self.beta)
            assert torch.all(torch.not_equal(batch_C, 0.0)), f'batch_C=0: something is probably off with the FEM setup'  # no C=0 allowed
            assert not torch.all(torch.eq(batch_dCdrho, 0.0)), f'batch_dCdrho all zeros: something is probably off with the FEM setup' # it's not allowed to have all zeros
           
            if self.config['lambda_comp'] > 0 and epoch >= self.config.get('start_comp', 0):
                # plot individual compliance fields
                if self.config['plot_grad_field'] and is_every_n_epochs_fulfilled(epoch, self.config, 'plot_every_n_epochs'):
                    if self.config['nx'] == 2:
                        grad_fields = einops.rearrange(batch_dCdrho[:, self.feni.sort_idcs], 'bz (y x) -> bz x y', x=self.feni.chosen_resolution[1]).detach().cpu().numpy()
                    elif self.config['nx'] == 3:
                        grad_fields = einops.rearrange(batch_dCdrho[:, self.feni.sort_idcs], 'bz (x y z) -> bz z y x', x=self.feni.chosen_resolution[0], y=self.feni.chosen_resolution[1]).detach().cpu().numpy()
                    self.mpm.plot(self.plotter.plot_grad_field, 'plot_grad_field', kwargs_dict={'grad_fields': grad_fields, 'fig_label': f'comp grad field', 'attributes': batch_C.detach().cpu().numpy()})
            
                batch_C, batch_dCdrho = batch_C * self.config.get('scale_comp', 1.0), batch_dCdrho * self.config.get('scale_comp', 1.0)
                loss_dict[LossKey('comp')] = (batch_C.mean(), batch_dCdrho)
                field_dict[LossKey('comp')] = rho_batch
                
            if self.config['lambda_vol'] > 0:
                if self.config['plot_grad_field'] and is_every_n_epochs_fulfilled(epoch, self.config, 'plot_every_n_epochs'):
                    if self.config['nx'] == 2:
                        grad_fields = einops.rearrange(batch_dVdrho[:, self.feni.sort_idcs], 'bz (y x) -> bz x y', x=self.feni.chosen_resolution[1]).detach().cpu().numpy()
                    elif self.config['nx'] == 3:
                        grad_fields = einops.rearrange(batch_dVdrho[:, self.feni.sort_idcs], 'bz (x y z) -> bz z y x', x=self.feni.chosen_resolution[0], y=self.feni.chosen_resolution[1]).detach().cpu().numpy()
                    self.mpm.plot(self.plotter.plot_grad_field, 'plot_grad_field', kwargs_dict={'grad_fields': grad_fields, 'fig_label': f'vol grad field', 'attributes': batch_V.detach().cpu().numpy()})
                batch_V = batch_V * self.config.get('scale_vol', 1.0)
                batch_dVdrho = batch_dVdrho * self.config.get('scale_vol', 1.0)
                loss_dict[LossKey('vol')] = (batch_V.mean(), batch_dVdrho)
                field_dict[LossKey('vol')] = rho_batch
        
        ## compute scalar sublosses
        for key in self.scalar_loss_keys:
            if epoch < self.config.get('start_'+key.base_key, 0):
                loss_dict[key] = torch.tensor(0.0)
            else:
                loss_dict[key] = self.loss_dispatcher[key.base_key](z=z, z_corners=z_corners, p=self.p,
                        epoch=epoch, batch=batch, p_surface=self.p_surface, weights_surf_pts=self.weights_surf_pts, x_fem=x_fem)

        ## compute total loss
        loss = self.loss_calculator.compute_loss_and_save_sublosses(loss_dict)
        # When diversity is on but get_surface_pts failed (degenerate SDF), skip backward so we don't
        # push the model into collapse. Backward uses sum(loss_weighted_dict), so we must skip the call.
        skip_step = (
            epoch >= self.config.get('start_div', 0)
            and self.p_surface is None
            and LossKey('div') in self.all_loss_keys
        )
        if skip_step:
            self.logger.info('Surface points failed (epoch >= start_div) — skipping backward')
        self._last_step_skipped = skip_step  # so trainer can skip adaptive_update when we didn't step
        
        # do backward here directly, as we need x_field (skip when surface points failed)
        if not skip_step:
            if self.config.get('use_config', False):
                self.loss_calculator.backward_config(field_dict, self.model)
            else:
                self.loss_calculator.backward(field_dict)
        else:
            # No gradients: opt.step() will add zero to params
            pass

        return loss, self.loss_calculator.get_dicts()
    
    def log_to_wandb(self, epoch, await_all=False):
        with Timer.record('plot_helper.pool await async results'):
            ## Wait for plots to be ready, then log them
            while self.cur_plot_epoch <= epoch:
                if not self.mpm.are_plots_available_for_epoch(self.cur_plot_epoch):
                    ## no plots for this epoch; just log the current losses
                    try:
                        wandb.log(self.log_history_dict[self.cur_plot_epoch])
                    except Exception as e:
                        self.logger.warning(f"wandb.log failed (training continues): {e}")
                    del self.log_history_dict[self.cur_plot_epoch]
                    self.cur_plot_epoch += 1
                elif self.mpm.plots_ready_for_epoch(self.cur_plot_epoch):
                    ## plots are available and ready
                    try:
                        wandb.log(self.log_history_dict[self.cur_plot_epoch] | self.mpm.pop_results_dict(self.cur_plot_epoch))
                    except Exception as e:
                        self.logger.warning(f"wandb.log failed (training continues): {e}")
                    del self.log_history_dict[self.cur_plot_epoch]
                    self.cur_plot_epoch += 1
                elif await_all:
                    ## plots are not ready yet - wait for them
                    self.logger.debug(f'Waiting for plots for epoch {self.cur_plot_epoch}')
                    time.sleep(1)
                else:
                    # print(f'Waiting for plots for epoch {cur_plot_epoch}')
                    break

    def init_loss_dispatcher(self):
        """
        Initializes the loss dispatcher, which is a dictionary mapping LossKey.base_key to loss functions.
        The loss functions are partial functions of the actual loss functions, with the model, logger, etc. already set.
        All the parameter set here are static and do not change during training.
        Dynamic parameters (e.g. z, p_surface, epoch) are passed to the (partial) loss functions at train time.
        At train time all (partial) loss functions are called with the same parameters, i.e all losses have the same signature.
        """
        loss_dispatcher = {
            # ginn losses
            # local
            'env': partial(loss_env, netp=self.netp, p_sampler=self.problem, level_set=self.config['level_set'], nf_is_density=self.config['nf_is_density']),
            'obst': partial(loss_obst, netp=self.netp, p_sampler=self.problem, level_set=self.config['level_set'], nf_is_density=self.config['nf_is_density']),
            'if': partial(loss_if, netp=self.netp, p_sampler=self.problem, level_set=self.config['level_set']),
            'if_normal': partial(loss_if_normal, netp=self.netp, p_sampler=self.problem, ginn_bsize=self.config['ginn_bsize'], loss_scale=self.config.get('scale_if_normal', 1), nf_is_density=self.config['nf_is_density']),
            'cuboid_rule': partial(loss_cuboid_rule, netp=self.netp, p_sampler=self.problem, level_set=self.config['level_set'], nf_is_density=self.config['nf_is_density']),
            # NEW: Rule-based cuboid primitive loss with stratified sampling and boundary sharpening
            'cuboid_primitive': partial(loss_cuboid_primitive, netp=self.netp, bounds=self.problem.bounds,
                                        n_inside=self.config.get('cuboid_n_inside', 2000),
                                        n_near_faces=self.config.get('cuboid_n_near_faces', 1000),
                                        n_corners=self.config.get('cuboid_n_corners', 500),
                                        n_far=self.config.get('cuboid_n_far', 500),
                                        n_boundary=self.config.get('cuboid_n_boundary', 1000),
                                        safety_margin=self.config.get('cuboid_safety_margin', 0.1),
                                        boundary_weight=self.config.get('cuboid_boundary_weight', 0.5)),
            
            # global
            'eikonal': partial(loss_eikonal, p_sampler=self.problem, netp=self.netp, scale_eikonal=self.config.get('scale_eikonal', 1), nf_is_density=self.config['nf_is_density']),
            'scc': partial(loss_scc, ph_manager=self.ph_manager),
            # NEW: Latent structure regularization (variance + magnitude + diversity)
            'latent_structure': partial(loss_latent_structure, netp=self.netp, p_sampler=self.problem,
                                        min_variance=self.config.get('latent_min_variance', 0.1),
                                        max_magnitude=self.config.get('latent_max_magnitude', 3.0),
                                        diversity_weight=self.config.get('latent_diversity_weight', 1.0),
                                        n_diversity_pts=self.config.get('latent_n_diversity_pts', 100)),
            'curv': partial(loss_curv, netp=self.netp, logger=self.logger, \
                            max_curv=self.config['max_curv'], device=torch.get_default_device(), \
                            loss_scale=self.config.get('scale_curv', 1.0),
                            curvature_expression=self.config['curvature_expression'], \
                            strain_curvature_clip_max=self.config['strain_curvature_clip_max'], \
                            curvature_use_gradnorm_weights=self.config.get('curvature_use_gradnorm_weights', False), \
                            curvature_after_5000_epochs=self.config.get('curvature_after_5000_epochs', False),
                            curvature_pts_source=self.config['curvature_pts_source']),

            'div': partial(loss_div, netp=self.netp, logger=self.logger, p_sampler=self.problem, \
                           max_div=self.config['max_div'], ginn_bsize=self.config['latent_sampling']['ginn_bsize'], \
                           diversity_pts_source=self.config['diversity_pts_source'], \
                           div_norm_order=self.config['div_norm_order'], div_neighbor_agg_fn=self.config['div_neighbor_agg_fn'],
                           leinster_temp=self.config.get('leinster_temp', None), leinster_q=self.config.get('leinster_q', None),
                           loss_scale=self.config.get('scale_div', 1.0)),
            'chamfer_div': partial(loss_chamfer_div, netp=self.netp, chamfer_p=self.config.get('chamfer_p', 1), 
                                   max_div=self.config['max_div'], chamfer_div_eps=self.config.get('chamfer_div_eps', 1e-6), 
                                   loss_scale=self.config.get('scale_chamfer_div', 1), subsample=self.config.get('chamfer_div_subsample', None)),
            
            # New diversity losses for generative training
            'shape_diversity': partial(loss_shape_diversity, 
                                       diversity_type=self.config.get('diversity_type', 'chamfer'),
                                       aggregation=self.config.get('diversity_aggregation', 'min'),
                                       max_diversity=self.config.get('max_diversity', None),
                                       loss_scale=self.config.get('scale_shape_diversity', 1.0)),
            
            # Connectivity loss (Morse theory) for generative training
            'connectivity': partial(loss_connectivity,
                                    netp=self.netp,
                                    bounds=self.problem.bounds,
                                    cached_connectivity_loss=self.cached_connectivity_loss,
                                    loss_scale=self.config.get('scale_connectivity', 1.0)),
           
            # data losses
            'lip': partial(loss_lip, netp=self.netp),
            'dirichlet': partial(loss_dirichlet, netp=self.netp, p_sampler=self.problem),
            'data': partial(loss_data, netp=self.netp),
            # null loss for disabling the objective loss
            'null': partial(loss_null, netp=self.netp),
            
            'rotsym': partial(loss_rotation_symmetric, netp=self.netp, p_sampler=self.problem, n_cycles=self.config['problem']['rotation_n_cycles']),
            
        }
        if getattr(self, 'conditional_prior', None) is not None:
            loss_dispatcher['prior'] = partial(
                loss_prior,
                conditional_prior=self.conditional_prior,
                nz_base=self.nz_base_prior,
                c_dim=self.c_dim_prior,
            )
        return loss_dispatcher