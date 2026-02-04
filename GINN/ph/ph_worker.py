"""
Torch-free entry points for PH pool workers (Windows multiprocessing spawn).
Workers must not import torch/CUDA or training hangs. This module only uses numpy and cripser.
"""
import numpy as np
import cripser


def calc_3d_ph_4d_array(Y_slice: np.ndarray, maxdim: int = 0) -> np.ndarray:
    """Compute persistent homology for plotting. Used by PHManager.calc_ph()."""
    return cripser.computePH(Y_slice, maxdim=maxdim)


def _filter_PH_iso(PHd: np.ndarray, lengthsd: np.ndarray, iso: float):
    is_iso = (PHd[:, 1] < -iso) & (PHd[:, 2] > iso)
    return PHd[is_iso], lengthsd[is_iso]


def _filter_PH_inf_death(PHd: np.ndarray, lengthsd: np.ndarray):
    is_iso = PHd[:, 2] < 1.0e100
    return PHd[is_iso], lengthsd[is_iso]


def _target_betti_at(target_Betti, dim: int):
    """Support TARGET_Betti as int or sequence."""
    if hasattr(target_Betti, "__getitem__"):
        return target_Betti[dim]
    return target_Betti


def calc_3d_ph_4d_array_full(
    Y_slice: np.ndarray,
    slice_idx: int,
    maxdim: int,
    z: np.ndarray,
    TARGET_Betti,
    iso: float,
    nx: int,
    ph_loss_super0_points: bool,
    ph_loss_sub0: bool,
    HOLE_LEVEL: float,
):
    """
    Compute PH and return (x_list, z_list, loss_key_list, loss_spec_list).
    loss_spec_list items are (func_name, kwargs) so the main process can
    resolve loss functions without the worker importing torch.
    """
    ph = cripser.computePH(Y_slice, maxdim=maxdim)
    return _slice_to_specs(
        z, ph, slice_idx, maxdim, TARGET_Betti, iso, nx,
        ph_loss_super0_points, ph_loss_sub0, HOLE_LEVEL,
    )


def _slice_to_specs(
    z: np.ndarray,
    PH: np.ndarray,
    slice_idx: int,
    maxdim: int,
    target_Betti,
    iso: float,
    nx: int,
    ph_loss_super0_points: bool,
    ph_loss_sub0: bool,
    hole_level: float,
):
    x_list = []
    z_list = []
    loss_key_list = []
    loss_spec_list = []  # (func_name, kwargs) for main to resolve

    for DIM in range(0, maxdim + 1):
        TARGET_Betti = _target_betti_at(target_Betti, DIM)
        PH_dim = PH[PH[:, 0] == DIM]
        lifetime = PH_dim[:, 2] - PH_dim[:, 1]
        sorted_ids = np.argsort(lifetime)[::-1]
        PH_dim = PH_dim[sorted_ids]
        lifetime = lifetime[sorted_ids]

        if DIM == 0:
            PH_filter, lifetime_filter = _filter_PH_iso(PH_dim, lifetime, iso)
            death_isos = PH_filter[:, 6 : 6 + nx].astype(int)
            x_idcs = death_isos[TARGET_Betti:, :]
            z_in = np.repeat(z[slice_idx : slice_idx + 1], repeats=x_idcs.shape[0], axis=0)
            x_list.append(x_idcs)
            z_list.append(z_in)
            loss_key_list.append("loss_scc")
            loss_spec_list.append(("loss_scc_dim_0", {"iso": iso}))

            if ph_loss_super0_points:
                raise NotImplementedError("Super loss outdated but kept for possible future use")
            if ph_loss_sub0:
                raise NotImplementedError("Sub loss outdated but kept for possible future use")

        elif DIM == 1:
            PH_filter, lifetime_filter = _filter_PH_inf_death(PH_dim, lifetime)
            death_isos = PH_filter[:, 6 : 6 + nx].astype(int)
            birth_isos = PH_filter[:, 3 : 3 + nx].astype(int)

            x_idcs = birth_isos[0:TARGET_Betti, :]
            z_in = np.repeat(z[slice_idx : slice_idx + 1], repeats=x_idcs.shape[0], axis=0)
            x_list.append(x_idcs)
            z_list.append(z_in)
            loss_key_list.append("loss_scc")
            loss_spec_list.append(("loss_scc_dim_1_neg", {"hole_level": hole_level}))

            x_idcs = death_isos[0:TARGET_Betti, :]
            z_in = np.repeat(z[slice_idx : slice_idx + 1], repeats=x_idcs.shape[0], axis=0)
            x_list.append(x_idcs)
            z_list.append(z_in)
            loss_key_list.append("loss_scc")
            loss_spec_list.append(("loss_scc_dim_1_pos", {"hole_level": hole_level}))

            PH_filter, lifetime_filter = _filter_PH_iso(
                PH_filter[TARGET_Betti:, :], lifetime_filter[TARGET_Betti:], iso
            )
            death_isos = PH_filter[:, 6 : 6 + nx].astype(int)
            x_idcs = death_isos[:, :]
            z_in = np.repeat(z[slice_idx : slice_idx + 1], repeats=x_idcs.shape[0], axis=0)
            x_list.append(x_idcs)
            z_list.append(z_in)
            loss_key_list.append("loss_scc")
            loss_spec_list.append(("loss_scc_dim_1", {"iso": iso}))

        elif DIM == 2:
            raise NotImplementedError("DIM 2 loss not implemented")

    return x_list, z_list, loss_key_list, loss_spec_list
