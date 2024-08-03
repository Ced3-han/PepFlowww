from pepflow.modules.so3.utils import *

import logging
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# Return angle of rotation. SO(3) to R^+
def Omega(R): return torch.arccos((torch.diagonal(R, dim1=-2, dim2=-1).sum(axis=-1)-1)/2)

# Power series expansion in the IGSO3 density.
def f_igso3(omega, t, L=500):
    ls = torch.arange(L)[None]  # of shape [1, L]
    return ((2*ls + 1) * torch.exp(-ls*(ls+1)*t/2) *
             torch.sin(omega[:, None]*(ls+1/2)) / torch.sin(omega[:, None]/2)).sum(dim=-1)

# IGSO3(Rt; I_3, t), density with respect to the volume form on SO(3) 
def igso3_density(Rt, t, L=500): return f_igso3(Omega(Rt), t, L)

# Marginal density of rotation angle for uniform density on SO(3)
def angle_density_unif(omega):
    return (1-torch.cos(omega))/np.pi

# Normal sample in tangent space at R0
def tangent_gaussian(R0): return torch.einsum('...ij,...jk->...ik', R0, hat(torch.randn(R0.shape[0], 3)))

def centered_gaussian(num_batch, num_res, device='cpu'):
    # torch.manual_seed(0)
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def uniform_so3(num_batch, num_res, device='cpu'):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

class SO3LookupCache:
    def __init__(
        self,
        cache_dir: str,
        cache_file: str,
        overwrite: bool = False,
    ) -> None:
        """
        Auxiliary class for handling storage / loading of SO(3) lookup tables in npz format.

        Args:
            cache_dir: Path to the cache directory.
            cache_file: Basic file name of the cache file.
            overwrite: Whether existing cache files should be overwritten if requested.
        """
        if not cache_file.endswith(".npz"):
            raise ValueError("Filename should have '.npz' extension.")
        self.cache_file = cache_file
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, cache_file)
        self.overwrite = overwrite

    @property
    def path_exists(self) -> bool:
        return os.path.exists(self.cache_path)

    @property
    def dir_exists(self) -> bool:
        return os.path.exists(self.cache_dir)

    def delete_cache(self) -> None:
        """
        Delete the cache file.
        """
        if self.path_exists:
            os.remove(self.cache_path)

    def load_cache(self) -> Dict[str, torch.Tensor]:
        """
        Load data from the cache file.

        Returns:
            Dictionary of loaded data tensors.
        """
        if self.path_exists:
            # Load data and convert to torch tensors.
            npz_data = np.load(self.cache_path)
            torch_dict = {f: torch.from_numpy(npz_data[f]) for f in npz_data.files}
            logger.info(f"Data loaded from {self.cache_path}")
            return torch_dict
        else:
            raise ValueError(f"No cache data found at {self.cache_path}.")

    def save_cache(self, data: Dict[str, torch.Tensor]) -> None:
        """
        Save a dictionary of tensors to the cache file. If overwrite is set to True, an existing
        file is overwritten, otherwise a warning is raised and the file is not modified.

        Args:
            data: Dictionary of tensors that should be saved to the cache.
        """
        if not self.dir_exists:
            os.makedirs(self.cache_dir)

        if self.path_exists:
            if self.overwrite:
                logger.info("Overwriting cache ...")
                self.delete_cache()
            else:
                logger.warn(
                    f"Cache at {self.cache_path} exits and overwriting disabled. Doing nothing."
                )
        else:
            # Move everything to CPU and numpy and store.
            logger.info(f"Data saved to {self.cache_path}")
            numpy_dict = {k: v.detach().cpu().numpy() for k, v in data.items()}
            np.savez(self.cache_path, **numpy_dict)


class BaseSampleSO3(nn.Module):
    so3_type: str = "base"  # cache basename

    def __init__(
        self,
        num_omega: int,
        sigma_grid: torch.Tensor,
        omega_exponent: int = 3,
        tol: float = 1e-7,
        interpolate: bool = True,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
        device: str = 'cpu',
    ) -> None:
        """
        Base torch.nn module for sampling rotations from the IGSO(3) distribution. Samples are
        created by uniformly sampling a rotation axis and using inverse transform sampling for
        the angles. The latter uses the associated SO(3) cumulative probability distribution
        function (CDF) and a uniform distribution [0,1] as described in [#leach2022_1]_. CDF values
        are obtained by numerically integrating the probability distribution evaluated on a grid of
        angles and noise levels and stored in a lookup table. Linear interpolation is used to
        approximate continuos sampling of the function. Angles are discretized in an interval [0,pi]
        and the grid can be squashed to have higher resolutions at low angles by taking different
        powers. Since sampling relies on tabulated values of the CDF and indexing in the form of
        `torch.bucketize`, gradients are not supported.

        Args:
            num_omega (int): Number of discrete angles used for generating the lookup table.
            sigma_grid (torch.Tensor): Grid of IGSO3 std devs.
            omega_exponent (int, optional): Make the angle grid denser for smaller angles by taking
              its power with the provided number. Defaults to 3.
            tol (float, optional): Small value for numerical stability. Defaults to 1e-7.
            interpolate (bool, optional): If enables, perform linear interpolation of the angle CDF
              to sample angles. Otherwise the closest tabulated point is returned. Defaults to True.
            cache_dir: Path to an optional cache directory. If set to None, lookup tables are
              computed on the fly.
            overwrite_cache: If set to true, existing cache files are overwritten. Can be used for
              updating stale caches.

        References
        ----------
        .. [#leach2022_1] Leach, Schmon, Degiacomi, Willcocks:
           Denoising diffusion probabilistic models on so (3) for rotational alignment.
           ICLR 2022 Workshop on Geometrical and Topological Representation Learning. 2022.
        """
        super().__init__()
        self.num_omega = num_omega
        self.omega_exponent = omega_exponent
        self.tol = tol
        self.interpolate = interpolate
        self.device = device
        self.register_buffer("sigma_grid", sigma_grid, persistent=False)

        # Generate / load lookups and store in non-persistent buffers.
        omega_grid, cdf_igso3 = self._setup_lookup(sigma_grid, cache_dir, overwrite_cache)
        self.register_buffer("omega_grid", omega_grid, persistent=False)
        self.register_buffer("cdf_igso3", cdf_igso3, persistent=False)

    def _setup_lookup(
        self,
        sigma_grid: torch.Tensor,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Master function for setting up the lookup tables. These can either be loaded from a npz
        cache file or computed on the fly. Lookup tables will always be created and stored in double
        precision. Casting to the target dtype is done at the end of the function.

        Args:
            sigma_grid: Grid of sigma values used for computing the lookup tables.
            cache_dir: Path to the cache directory.
            overwrite_cache: If set to true, an existing cache is overwritten. Can be used for
              updating stale caches.

        Returns:
            Grid of angle values and SO(3) cumulative distribution function.
        """
        if cache_dir is not None:
            cache_name = self._get_cache_name()
            cache = SO3LookupCache(cache_dir, cache_name, overwrite=True)

            # If cache dir is provided, check whether the necessary cache exists and whether it
            # should be overwritten.
            if cache.path_exists and not overwrite_cache:
                # Load data from cache.
                cache_data = cache.load_cache()
                omega_grid = cache_data["omega_grid"]
                cdf_igso3 = cache_data["cdf_igso3"]
            else:
                # Store data in cache (overwrite if requested).
                omega_grid, cdf_igso3 = self._generate_lookup(sigma_grid)
                cache.save_cache({"omega_grid": omega_grid, "cdf_igso3": cdf_igso3})
        else:
            # Other wise just generate the tables.
            omega_grid, cdf_igso3 = self._generate_lookup(sigma_grid)

        return omega_grid.to(sigma_grid.dtype), cdf_igso3.to(sigma_grid.dtype)

    def _get_cache_name(self) -> str:
        """
        Auxiliary function for determining the cache file name based on the parameters (sigma,
        omega, l, etc.) used for generating the lookup tables.

        Returns:
            Base name of the cache file.
        """
        cache_name = "cache_{:s}_s{:04.3f}-{:04.3f}-{:d}_o{:d}-{:d}.npz".format(
            self.so3_type,
            torch.min(self.sigma_grid).cpu().item(),
            torch.max(self.sigma_grid).cpu().item(),
            self.sigma_grid.shape[0],
            self.num_omega,
            self.omega_exponent,
        )
        return cache_name

    def get_sigma_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous sigmas to the indices of the closest tabulated values.

        Args:
            sigma (torch.Tensor): IGSO3 std devs.

        Returns:
            torch.Tensor: Index tensor mapping the provided sigma values to the internal lookup
              table.
        """
        return torch.bucketize(sigma, self.sigma_grid)

    def expansion_function(
        self, omega_grid: torch.Tensor, sigma_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Function for generating the angle probability distribution. Should return a 2D tensor with
        values for the std dev at the first dimension (rows) and angles at the second
        (columns).

        Args:
            omega_grid (torch.Tensor): Grid of angle values.
            sigma_grid (torch.Tensor): IGSO3 std devs.

        Returns:
            torch.Tensor: Distribution for angles discretized on a 2D grid.
        """
        raise NotImplementedError

    @torch.no_grad()
    def _generate_lookup(self, sigma_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the lookup table for sampling from the target SO(3) CDF. The table is 2D, with the
        rows corresponding to different sigma values and the columns with angles computed on a grid.
        Variance is scaled by a factor of 1/2 to account for the deacceleration of time in the
        diffusion process due to the choice of SO(3) basis and guarantee time-reversibility (see
        appendix E.3 in [#yim2023_2]_). The returned tables are double precision and will be cast
        to the target dtype in `_setup_lookup`.

        Args:
            sigma_grid (torch.Tensor): Grid of IGSO3 std devs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the grid used to compute the angles
              and the associated lookup table.

        References
        ----------
        .. [#yim2023_2] Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, Jaakkola:
           SE(3) diffusion model with application to protein backbone generation.
           arXiv preprint arXiv:2302.02277. 2023.
        """

        current_device = sigma_grid.device
        sigma_grid_tmp = sigma_grid.to(torch.float64)

        # If cuda is available, initialize everything on GPU.
        # Even if Pytorch Lightning usually handles GPU allocation after initialization, this is
        # required to initialize the module in GPU reducing the initializaiton time by orders of magnitude.
        if torch.cuda.is_available():
            sigma_grid_tmp = sigma_grid_tmp.to(device=self.device)

        # Set up grid for angle resolution. Convert to double precision for better handling of numerics.
        omega_grid = torch.linspace(0.0, 1, self.num_omega + 1).to(sigma_grid_tmp)

        # If requested, increase sample density for lower values
        omega_grid = omega_grid**self.omega_exponent

        omega_grid = omega_grid * np.pi

        # Compute the expansion for all omegas and sigmas.
        pdf_igso3 = self.expansion_function(omega_grid, sigma_grid_tmp)

        # Apply the pre-factor from USO(3).
        pdf_igso3 = pdf_igso3 * (1.0 - torch.cos(omega_grid)) / np.pi

        # Compute the cumulative probability distribution.
        cdf_igso3 = integrate_trapezoid_cumulative(pdf_igso3, omega_grid)
        # Normalize integral area to 1.
        cdf_igso3 = cdf_igso3 / cdf_igso3[:, -1][:, None]

        # Move back to original device.
        cdf_igso3 = cdf_igso3.to(device=current_device)
        omega_grid = omega_grid.to(device=current_device)

        return omega_grid[1:].to(sigma_grid.dtype), cdf_igso3.to(sigma_grid.dtype)
    
    def sample(self, sigma: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the target SO(3) distribution by sampling a rotation axis angle,
        which are then combined into a rotation vector and transformed into the corresponding
        rotation matrix via an exponential map.

        Args:
            sigma_indices (torch.Tensor): Indices of the IGSO3 std devs for which to take samples.
            num_samples (int): Number of angle samples to take for each std dev

        Returns:
            torch.Tensor: Sampled rotations in matrix representation with dimensions
              [num_sigma x num_samples x 3 x 3].
        """
        # torch.manual_seed(0)

        vectors = self.sample_vector(sigma.shape[0], num_samples)
        angles = self.sample_angle(sigma, num_samples)

        # Do postprocessing on angles.
        angles = self._process_angles(sigma, angles)

        rotation_vectors = vectors * angles[..., None]

        rotation_matrices = rotvec_to_rotmat(rotation_vectors, tol=self.tol)
        return rotation_matrices

    def _process_angles(self, sigma: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function for performing additional processing steps on the sampled angles. One
        example would be to ensure sampled angles are 0 for a std dev of 0 for IGSO(3).

        Args:
            sigma (torch.Tensor): Current values of sigma.
            angles (torch.Tensor): Sampled angles.

        Returns:
            torch.Tensor: Processed sampled angles.
        """
        return angles

    def sample_vector(self, num_sigma: int, num_samples: int) -> torch.Tensor:
        """
        Uniformly sample rotation axis for constructing the overall rotation.

        Args:
            num_sigma (int): Number of samples to draw for each std dev.
            num_samples (int): Number of angle samples to take for each std dev.

        Returns:
            torch.Tensor: Batch of rotation axes with dimensions [num_sigma x num_samples x 3].
        """
        vectors = torch.randn(num_sigma, num_samples, 3, device=self.sigma_grid.device)
        vectors = vectors / torch.norm(vectors, dim=2, keepdim=True)
        return vectors

    def sample_angle(self, sigma: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Create a series of samples from the IGSO(3) angle distribution.

        Args:
            sigma_indices (torch.Tensor): Indices of the IGSO3 std deves for which to
              take samples.
            num_samples (int): Number of angle samples to take for each std dev.

        Returns:
            torch.Tensor: Collected samples, will have the dimension [num_sigma x num_samples].
        """
        # Convert sigmas to respective indices for lookup table.
        sigma_indices = self.get_sigma_idx(sigma)
        # Get relevant sigma slices from stored CDFs.
        cdf_tmp = self.cdf_igso3[sigma_indices, :]

        # Draw from uniform distribution.
        p_uniform = torch.rand((*sigma_indices.shape, *[num_samples]), device=sigma_indices.device)

        # Determine indices for CDF.
        idx_stop = torch.sum(cdf_tmp[..., None] < p_uniform[:, None, :], dim=1).long()
        idx_start = torch.clamp(idx_stop - 1, min=0)

        if not self.interpolate:
            omega = torch.gather(cdf_tmp, dim=1, index=idx_stop)
        else:
            # Get CDF values.
            cdf_start = torch.gather(cdf_tmp, dim=1, index=idx_start)
            cdf_stop = torch.gather(cdf_tmp, dim=1, index=idx_stop)

            # Compute weights for linear interpolation.
            cdf_delta = torch.clamp(cdf_stop - cdf_start, min=self.tol)
            cdf_weight = torch.clamp((p_uniform - cdf_start) / cdf_delta, min=0.0, max=1.0)

            # Get angle range for interpolation.
            omega_start = self.omega_grid[idx_start]
            omega_stop = self.omega_grid[idx_stop]

            # Interpolate.
            omega = torch.lerp(omega_start, omega_stop, cdf_weight)

        return omega


class SampleIGSO3(BaseSampleSO3):
    so3_type = "igso3"  # cache basename

    def __init__(
        self,
        num_omega: int,
        sigma_grid: torch.Tensor,
        omega_exponent: int = 3,
        tol: float = 1e-7,
        interpolate: bool = True,
        l_max: int = 1000,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
        device: str = 'cpu',
    ) -> None:
        """
        Module for sampling rotations from the IGSO(3) distribution using the explicit series
        expansion.  Samples are created using inverse transform sampling based on the associated
        cumulative probability distribution function (CDF) and a uniform distribution [0,1] as
        described in [#leach2022_2]_. CDF values are obtained by numerically integrating the
        probability distribution evaluated on a grid of angles and noise levels and stored in a
        lookup table.  Linear interpolation is used to approximate continuos sampling of the
        function. Angles are discretized in an interval [0,pi] and the grid can be squashed to have
        higher resolutions at low angles by taking different powers.
        Since sampling relies on tabulated values of the CDF and indexing in the form of
        `torch.bucketize`, gradients are not supported.

        Args:
            num_omega (int): Number of discrete angles used for generating the lookup table.
            sigma_grid (torch.Tensor): Grid of IGSO3 std devs.
            omega_exponent (int, optional): Make the angle grid denser for smaller angles by taking
              its power with the provided number. Defaults to 3.
            tol (float, optional): Small value for numerical stability. Defaults to 1e-7.
            interpolate (bool, optional): If enables, perform linear interpolation of the angle CDF
              to sample angles. Otherwise the closest tabulated point is returned. Defaults to True.
            l_max (int, optional): Maximum number of terms used in the series expansion.
            cache_dir: Path to an optional cache directory. If set to None, lookup tables are
              computed on the fly.
            overwrite_cache: If set to true, existing cache files are overwritten. Can be used for
              updating stale caches.

        References
        ----------
        .. [#leach2022_2] Leach, Schmon, Degiacomi, Willcocks:
           Denoising diffusion probabilistic models on so (3) for rotational alignment.
           ICLR 2022 Workshop on Geometrical and Topological Representation Learning. 2022.
        """
        self.l_max = l_max
        super().__init__(
            num_omega=num_omega,
            sigma_grid=sigma_grid,
            omega_exponent=omega_exponent,
            tol=tol,
            interpolate=interpolate,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            device=device,
        )

    def _get_cache_name(self) -> str:
        """
        Auxiliary function for determining the cache file name based on the parameters (sigma,
        omega, l, etc.) used for generating the lookup tables.

        Returns:
            Base name of the cache file.
        """
        cache_name = "cache_{:s}_s{:04.3f}-{:04.3f}-{:d}_l{:d}_o{:d}-{:d}.npz".format(
            self.so3_type,
            torch.min(self.sigma_grid).cpu().item(),
            torch.max(self.sigma_grid).cpu().item(),
            self.sigma_grid.shape[0],
            self.l_max,
            self.num_omega,
            self.omega_exponent,
        )
        return cache_name

    def expansion_function(
        self,
        omega_grid: torch.Tensor,
        sigma_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Use the truncated expansion of the IGSO(3) probability function to generate the lookup table.

        Args:
            omega_grid (torch.Tensor): Grid of angle values.
            sigma_grid (torch.Tensor): Grid of IGSO3 std devs.

        Returns:
            torch.Tensor: IGSO(3) distribution for angles discretized on a 2D grid.
        """
        return generate_igso3_lookup_table(omega_grid, sigma_grid, l_max=self.l_max, tol=self.tol)

    def _process_angles(self, sigma: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Ensure sampled angles are 0 for small noise levels in IGSO(3). (Series expansion gives
        uniform probability distribution.)

        Args:
            sigma (torch.Tensor): Current values of sigma.
            angles (torch.Tensor): Sampled angles.

        Returns:
            torch.Tensor: Processed sampled angles.
        """
        angles = torch.where(
            sigma[..., None] < self.tol,
            torch.zeros_like(angles),
            angles,
        )
        return angles



class SampleUSO3:
    def sample(self, sigma: torch.Tensor, num_samples: int):
        return torch.tensor(Rotation.random(num_samples).as_matrix(), dtype=torch.float32)

# class SampleUSO3(BaseSampleSO3):
#     so3_type = "uso3"  # cache basename

#     def __init__(
#         self,
#         num_omega: int,
#         sigma_grid: torch.Tensor,
#         omega_exponent: int = 3,
#         tol: float = 1e-7,
#         interpolate: bool = True,
#         cache_dir: Optional[str] = None,
#         overwrite_cache: bool = False,
#     ) -> None:
#         """
#         Module for sampling rotations from the USO(3) distribution. Can be used to generate initial
#         unbiased samples in the reverse process.  Samples are created using inverse transform
#         sampling based on the associated cumulative probability distribution function (CDF) and a
#         uniform distribution [0,1] as described in [#leach2022_4]_. CDF values are obtained by
#         numerically integrating the probability distribution evaluated on a grid of angles and noise
#         levels and stored in a lookup table.  Linear interpolation is used to approximate continuos
#         sampling of the function. Angles are discretized in an interval [0,pi] and the grid can be
#         squashed to have higher resolutions at low angles by taking different powers.
#         Since sampling relies on tabulated values of the CDF and indexing in the form of
#         `torch.bucketize`, gradients are not supported.

#         Args:
#             num_omega (int): Number of discrete angles used for generating the lookup table.
#             sigma_grid (torch.Tensor): Grid of IGSO3 std devs.
#             omega_exponent (int, optional): Make the angle grid denser for smaller angles by taking
#               its power with the provided number. Defaults to 3.
#             tol (float, optional): Small value for numerical stability. Defaults to 1e-7.
#             interpolate (bool, optional): If enables, perform linear interpolation of the angle CDF
#               to sample angles. Otherwise the closest tabulated point is returned. Defaults to True.
#             cache_dir: Path to an optional cache directory. If set to None, lookup tables are
#               computed on the fly.
#             overwrite_cache: If set to true, existing cache files are overwritten. Can be used for
#               updating stale caches.

#         References
#         ----------
#         .. [#leach2022_4] Leach, Schmon, Degiacomi, Willcocks:
#            Denoising diffusion probabilistic models on so (3) for rotational alignment.
#            ICLR 2022 Workshop on Geometrical and Topological Representation Learning. 2022.
#         """
#         super().__init__(
#             num_omega=num_omega,
#             sigma_grid=sigma_grid,
#             omega_exponent=omega_exponent,
#             tol=tol,
#             interpolate=interpolate,
#             cache_dir=cache_dir,
#             overwrite_cache=overwrite_cache,
#         )

#     def get_sigma_idx(self, sigma: torch.Tensor) -> torch.Tensor:
#         return torch.zeros_like(sigma).long()

#     def sample_shape(self, num_sigma: int, num_samples: int) -> torch.Tensor:
#         dummy_sigma = torch.zeros(num_sigma, device=self.sigma_grid.device)
#         return self.sample(dummy_sigma, num_samples)

#     def expansion_function(
#         self,
#         omega_grid: torch.Tensor,
#         sigma_grid: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         The probability density function of the uniform SO(3) distribution is the cosine scaling
#         term (1-cos(omega))/pi which is applied automatically during sampling. This means, it is
#         sufficient to return a tensor of ones to create the correct USO(3) lookup table.

#         Args:
#             omega_grid (torch.Tensor): Grid of angle values.
#             sigma_grid (torch.Tensor): Grid of IGSO3 std devs.

#         Returns:
#             torch.Tensor: USO(3) distribution for angles discretized on a 2D grid.
#         """
#         return torch.ones(1, omega_grid.shape[0], device=omega_grid.device)


@torch.no_grad()
def integrate_trapezoid_cumulative(f_grid: torch.Tensor, x_grid: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary function for numerically integrating a discretized 1D function using the trapezoid
    rule. This is mainly used for computing the cumulative probability distributions for sampling
    from the IGSO(3) distribution. Works on a single 1D grid or a batch of grids.

    Args:
        f_grid (torch.Tensor): Discretized function values.
        x_grid (torch.Tensor): Discretized input values.

    Returns:
        torch.Tensor: Integrated function (not normalized).
    """
    f_sum = f_grid[..., :-1] + f_grid[..., 1:]
    delta_x = torch.diff(x_grid, dim=-1)
    integral = torch.cumsum((f_sum * delta_x[None, :]) / 2.0, dim=-1)
    return integral


def uniform_so3_density(omega: torch.Tensor) -> torch.Tensor:
    """
    Compute the density over the uniform angle distribution in SO(3).

    Args:
        omega: Angles in radians.

    Returns:
        Uniform distribution density.
    """
    return (1.0 - torch.cos(omega)) / np.pi


def igso3_expansion(
    omega: torch.Tensor, sigma: torch.Tensor, l_grid: torch.Tensor, tol=1e-7
) -> torch.Tensor:
    """
    Compute the IGSO(3) angle probability distribution function for pairs of angles and std dev
    levels. The expansion is computed using a grid of expansion orders ranging from 0 to l_max.

    This function approximates the power series in equation 5 of [#yim2023_3]_. With this
    parameterization, IGSO(3) agrees with the Brownian motion on SO(3) with t=sigma^2.

    Args:
        omega: Values of angles (1D tensor).
        sigma: Values of std dev of IGSO3 distribution (1D tensor of same shape as `omega`).
        l_grid: Tensor containing expansion orders (0 to l_max).
        tol: Small offset for numerical stability.

    Returns:
        IGSO(3) angle distribution function (without pre-factor for uniform SO(3) distribution).

    References
    ----------
    .. [#yim2023_3] Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, Jaakkola:
        SE(3) diffusion model with application to protein backbone generation.
        arXiv preprint arXiv:2302.02277. 2023.
    """
    # Pre-compute sine in denominator and clamp for stability.
    denom_sin = torch.sin(0.5 * omega)

    # Pre-compute terms that rely only on expansion orders.
    l_fac_1 = 2.0 * l_grid + 1.0
    l_fac_2 = -l_grid * (l_grid + 1.0)

    # Pre-compute numerator of expansion which only depends on angles.
    numerator_sin = torch.sin((l_grid[None, :] + 1 / 2) * omega[:, None])

    # Pre-compute exponential term with (2l+1) prefactor.
    exponential_term = l_fac_1[None, :] * torch.exp(l_fac_2[None, :] * sigma[:, None] ** 2 / 2)

    # Compute series expansion
    f_igso = torch.sum(exponential_term * numerator_sin, dim=1)
    # For small omega, accumulate limit of sine fraction instead:
    # lim[x->0] sin((l+1/2)x) / sin(x/2) = 2l + 1
    f_limw = torch.sum(exponential_term * l_fac_1[None, :], dim=1)

    # Finalize expansion. Offset for stability can be added since omega is [0,pi] and sin(omega/2)
    # is positive in this interval.
    f_igso = f_igso / (denom_sin + tol)

    # Replace values at small omega with limit.
    f_igso = torch.where(omega <= tol, f_limw, f_igso)

    # Remove remaining numerical problems
    f_igso = torch.where(
        torch.logical_or(torch.isinf(f_igso), torch.isnan(f_igso)), torch.zeros_like(f_igso), f_igso
    )

    return f_igso


def digso3_expansion(
    omega: torch.Tensor, sigma: torch.Tensor, l_grid: torch.Tensor, tol=1e-7
) -> torch.Tensor:
    """
    Compute the derivative of the IGSO(3) angle probability distribution function with respect to
    the angles for pairs of angles and std dev levels. As in `igso3_expansion` a grid is used for the
    expansion levels. Evaluates the derivative directly in order to avoid second derivatives during
    backpropagation.

    The derivative of the angle-dependent part is computed as:

    .. math ::
        \frac{\partial}{\partial \omega} \frac{\sin((l+\tfrac{1}{2})\omega)}{\sin(\tfrac{1}{2}\omega)} = \frac{l\sin((l+1)\omega) - (l+1)\sin(l\omega)}{1 - \cos(\omega)}

    (obtained via quotient rule + different trigonometric identities).

    Args:
        omega: Values of angles (1D tensor).
        sigma: Values of IGSO3 distribution std devs (1D tensor of same shape as `omega`).
        l_grid: Tensor containing expansion orders (0 to l_max).
        tol: Small offset for numerical stability.

    Returns:
        IGSO(3) angle distribution derivative (without pre-factor for uniform SO(3) distribution).
    """
    denom_cos = 1.0 - torch.cos(omega)

    l_fac_1 = 2.0 * l_grid + 1.0
    l_fac_2 = l_grid + 1.0
    l_fac_3 = -l_grid * l_fac_2

    # Pre-compute numerator of expansion which only depends on angles.
    numerator_sin = l_grid[None, :] * torch.sin(l_fac_2[None, :] * omega[:, None]) - l_fac_2[
        None, :
    ] * torch.sin(l_grid[None, :] * omega[:, None])

    # Compute series expansion
    df_igso = torch.sum(
        l_fac_1[None, :] * torch.exp(l_fac_3[None, :] * sigma[:, None] ** 2 / 2) * numerator_sin,
        dim=1,
    )

    # Finalize expansion. Offset for stability can be added since omega is [0,pi] and cosine term
    # is positive in this interval.
    df_igso = df_igso / (denom_cos + tol)

    # Replace values at small omega with limit (=0).
    df_igso = torch.where(omega <= tol, torch.zeros_like(df_igso), df_igso)

    # Remove remaining numerical problems
    df_igso = torch.where(
        torch.logical_or(torch.isinf(df_igso), torch.isnan(df_igso)),
        torch.zeros_like(df_igso),
        df_igso,
    )

    return df_igso


def dlog_igso3_expansion(
    omega: torch.Tensor, sigma: torch.Tensor, l_grid: torch.Tensor, tol=1e-7
) -> torch.Tensor:
    """
    Compute the derivative of the logarithm of the IGSO(3) angle distribution function for pairs of
    angles and std dev levels:

    .. math ::
        \frac{\partial}{\partial \omega} \log f(\omega) = \frac{\tfrac{\partial}{\partial \omega} f(\omega)}{f(\omega)}

    Required for SO(3) score computation.

    Args:
        omega: Values of angles (1D tensor).
        sigma: Values of IGSO3 std devs (1D tensor of same shape as `omega`).
        l_grid: Tensor containing expansion orders (0 to l_max).
        tol: Small offset for numerical stability.

    Returns:
        IGSO(3) angle distribution derivative (without pre-factor for uniform SO(3) distribution).
    """
    f_igso3 = igso3_expansion(omega, sigma, l_grid, tol=tol)
    df_igso3 = digso3_expansion(omega, sigma, l_grid, tol=tol)

    return df_igso3 / (f_igso3 + tol)


@torch.no_grad()
def generate_lookup_table(
    base_function: Callable,
    omega_grid: torch.Tensor,
    sigma_grid: torch.Tensor,
    l_max: int = 1000,
    tol: float = 1e-7,
):
    """
    Auxiliary function for generating a lookup table from IGSO(3) expansions and their derivatives.
    Takes a basic function and loops over different std dev levels.

    Args:
        base_function: Function used for setting up the lookup table.
        omega_grid: Grid of angle values ranging from [0,pi] (shape is[num_omega]).
        sigma_grid: Grid of IGSO3 std dev values (shape is [num_sigma]).
        l_max: Number of terms used in the series expansion.
        tol: Small value for numerical stability.

    Returns:
        Table of function values evaluated at different angles and std dev levels. The final shape is
        [num_sigma x num_omega].
    """
    # Generate grid of expansion orders.
    l_grid = torch.arange(l_max + 1, device=omega_grid.device).to(omega_grid.dtype)

    n_omega = len(omega_grid)
    n_sigma = len(sigma_grid)

    # Populate lookup table for different time frames.
    f_table = torch.zeros(n_sigma, n_omega, device=omega_grid.device, dtype=omega_grid.dtype)

    for eps_idx in tqdm(range(n_sigma), desc=f"Computing {base_function.__name__}"):
        f_table[eps_idx, :] = base_function(
            omega_grid,
            torch.ones_like(omega_grid) * sigma_grid[eps_idx],
            l_grid,
            tol=tol,
        )

    return f_table


def generate_igso3_lookup_table(
    omega_grid: torch.Tensor,
    sigma_grid: torch.Tensor,
    l_max: int = 1000,
    tol: float = 1e-7,
) -> torch.Tensor:
    """
    Generate a lookup table for the IGSO(3) probability distribution function of angles.

    Args:
        omega_grid: Grid of angle values ranging from [0,pi] (shape is[num_omega]).
        sigma_grid: Grid of IGSO3 std dev values (shape is [num_sigma]).
        l_max: Number of terms used in the series expansion.
        tol: Small value for numerical stability.

    Returns:
        Table of function values evaluated at different angles and std dev levels. The final shape is
        [num_sigma x num_omega].
    """
    f_igso = generate_lookup_table(
        base_function=igso3_expansion,
        omega_grid=omega_grid,
        sigma_grid=sigma_grid,
        l_max=l_max,
        tol=tol,
    )
    return f_igso


def generate_dlog_igso3_lookup_table(
    omega_grid: torch.Tensor,
    sigma_grid: torch.Tensor,
    l_max: int = 1000,
    tol: float = 1e-7,
) -> torch.Tensor:
    """
    Generate a lookup table for the derivative of the logarithm of the angular IGSO(3) probability
    distribution function. Used e.g. for computing scaling of SO(3) norms.

    Args:
        omega_grid: Grid of angle values ranging from [0,pi] (shape is[num_omega]).
        sigma_grid: Grid of IGSO3 std dev values (shape is [num_sigma]).
        l_max: Number of terms used in the series expansion.
        tol: Small value for numerical stability.

    Returns:
        Table of function values evaluated at different angles and std dev levels. The final shape is
        [num_sigma x num_omega].
    """
    dlog_igso = generate_lookup_table(
        base_function=dlog_igso3_expansion,
        omega_grid=omega_grid,
        sigma_grid=sigma_grid,
        l_max=l_max,
        tol=tol,
    )
    return dlog_igso


if __name__ == '__main__':
    sigma_grid = torch.linspace(0.1, 1.5, 1000)
    igso3 = SampleIGSO3(1000, sigma_grid, cache_dir='.cache')
    print(igso3.sample(torch.tensor([1.5]),4))

    uso3 = SampleUSO3(1000, sigma_grid, cache_dir='.cache')
    print(uso3.sample(torch.tensor([1.5]),4))