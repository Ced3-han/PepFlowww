import logging
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

import warnings
warnings.filterwarnings('ignore')
from einops import rearrange

from copy import copy
from scipy.spatial.transform import Rotation

def gen_random_rotmats(num_samples):
    return torch.from_numpy(np.stack([Rotation.random().as_matrix() for _ in range(num_samples)])).float()

def gen_random_rotvecs(num_samples):
    return torch.from_numpy(np.stack([Rotation.random().as_rotvec() for _ in range(num_samples)])).float()

def check_skew_sym(h):
    # check if matrix h is skew-symmetric
    SKEW_SYMMETRIC_TOL = 1e-4
    skew_sym_eq = torch.allclose(h, -h.permute(0, 2, 1), atol=SKEW_SYMMETRIC_TOL, rtol=0)
    if not skew_sym_eq:
        print('skew symmetric error', torch.abs(h+h.permute(0, 2, 1)).max())
        return False
    return True
    
    
def check_rot_mat(R):
    # check if matrix R is a rotation matrix
    # (N,3,3)
    ROT_MAT_TOL = 1e-4
    rot_eq = torch.allclose(torch.inverse(R), R.permute(0, 2, 1), atol=ROT_MAT_TOL, rtol=0)
    rot_det_eq = torch.allclose(torch.det(R), torch.ones_like(torch.det(R)), atol=ROT_MAT_TOL, rtol=0)    
    if not (rot_eq and rot_det_eq):
        return False
    return True

def scale_rotmat(
    rotation_matrix: torch.Tensor, scalar: torch.Tensor, tol: float = 1e-7
) -> torch.Tensor:
    """
    Scale rotation matrix. This is done by converting it to vector representation,
    scaling the length of the vector and converting back to matrix representation.

    Args:
        rotation_matrix: Rotation matrices.
        scalar: Scalar values used for scaling. Should have one fewer dimension than the
            rotation matrices for correct broadcasting.
        tol: Numerical offset for stability.

    Returns:
        Scaled rotation matrix.
    """
    # Check whether dimensions match.
    assert rotation_matrix.ndim - 1 == scalar.ndim
    scaled_rmat = rotvec_to_rotmat(rotmat_to_rotvec(rotation_matrix) * scalar, tol=tol)
    return scaled_rmat


def _broadcast_identity(target: torch.Tensor) -> torch.Tensor:
    """
    Generate a 3 by 3 identity matrix and broadcast it to a batch of target matrices.

    Args:
        target (torch.Tensor): Batch of target 3 by 3 matrices.

    Returns:
        torch.Tensor: 3 by 3 identity matrices in the shapes of the target.
    """
    id3 = torch.eye(3, device=target.device, dtype=target.dtype)
    id3 = torch.broadcast_to(id3, target.shape)
    return id3


def skew_matrix_exponential_map_axis_angle(
    angles: torch.Tensor, skew_matrices: torch.Tensor
) -> torch.Tensor:
    """
    Compute the matrix exponential of a rotation in axis-angle representation with the axis in skew
    matrix representation form. Maps the rotation from the lie group to the rotation matrix
    representation. Uses Rodrigues' formula instead of `torch.linalg.matrix_exp` for better
    computational performance:

    .. math::

        \exp(\theta \mathbf{K}) = \mathbf{I} + \sin(\theta) \mathbf{K} + [1 - \cos(\theta)] \mathbf{K}^2

    Args:
        angles (torch.Tensor): Batch of rotation angles.
        skew_matrices (torch.Tensor): Batch of rotation axes in skew matrix (lie so(3)) basis.

    Returns:
        torch.Tensor: Batch of corresponding rotation matrices.
    """
    # Set up identity matrix and broadcast.
    id3 = _broadcast_identity(skew_matrices)

    # Broadcast angle vector to right dimensions
    angles = angles[..., None, None]

    exp_skew = (
        id3
        + torch.sin(angles) * skew_matrices
        + (1.0 - torch.cos(angles))
        * torch.einsum("b...ik,b...kj->b...ij", skew_matrices, skew_matrices)
    )
    return exp_skew


def skew_matrix_exponential_map(
    angles: torch.Tensor, skew_matrices: torch.Tensor, tol=1e-7
) -> torch.Tensor:
    """
    Compute the matrix exponential of a rotation vector in skew matrix representation. Maps the
    rotation from the lie group to the rotation matrix representation. Uses the following form of
    Rodrigues' formula instead of `torch.linalg.matrix_exp` for better computational performance
    (in this case the skew matrix already contains the angle factor):

    .. math ::

        \exp(\mathbf{K}) = \mathbf{I} + \frac{\sin(\theta)}{\theta} \mathbf{K} + \frac{1-\cos(\theta)}{\theta^2} \mathbf{K}^2

    This form has the advantage, that Taylor expansions can be used for small angles (instead of
    having to compute the unit length axis by dividing the rotation vector by small angles):

    .. math ::

        \frac{\sin(\theta)}{\theta} \approx 1 - \frac{\theta^2}{6}
        \frac{1-\cos(\theta)}{\theta^2} \approx \frac{1}{2} - \frac{\theta^2}{24}

    Args:
        angles (torch.Tensor): Batch of rotation angles.
        skew_matrices (torch.Tensor): Batch of rotation axes in skew matrix (lie so(3)) basis.

    Returns:
        torch.Tensor: Batch of corresponding rotation matrices.
    """
    # Set up identity matrix and broadcast.
    id3 = _broadcast_identity(skew_matrices)

    # Broadcast angles and pre-compute square.
    angles = angles[..., None, None]
    angles_sq = angles.square()

    # Get standard terms.
    sin_coeff = torch.sin(angles) / angles
    cos_coeff = (1.0 - torch.cos(angles)) / angles_sq
    # Use second order Taylor expansion for values close to zero.
    sin_coeff_small = 1.0 - angles_sq / 6.0
    cos_coeff_small = 0.5 - angles_sq / 24.0

    mask_zero = torch.abs(angles) < tol
    sin_coeff = torch.where(mask_zero, sin_coeff_small, sin_coeff)
    cos_coeff = torch.where(mask_zero, cos_coeff_small, cos_coeff)

    # Compute matrix exponential using Rodrigues' formula.
    exp_skew = (
        id3
        + sin_coeff * skew_matrices
        + cos_coeff * torch.einsum("b...ik,b...kj->b...ij", skew_matrices, skew_matrices)
    )
    return exp_skew


def rotvec_to_rotmat(rotation_vectors: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:
    """
    Convert rotation vectors to rotation matrix representation. The length of the rotation vector
    is the angle of rotation, the unit vector the rotation axis.

    Args:
        rotation_vectors (torch.Tensor): Batch of rotation vectors.
        tol: small offset for numerical stability.

    Returns:
        torch.Tensor: Rotation in rotation matrix representation.
    """
    # Compute rotation angle as vector norm.
    rotation_angles = torch.norm(rotation_vectors, dim=-1)

    # Map axis to skew matrix basis.
    skew_matrices = vector_to_skew_matrix(rotation_vectors)

    # Compute rotation matrices via matrix exponential.
    rotation_matrices = skew_matrix_exponential_map(rotation_angles, skew_matrices, tol=tol)

    return rotation_matrices


def rotmat_to_rotvec(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of rotation matrices to rotation vectors (logarithmic map from SO(3) to so(3)).
    The standard logarithmic map can be derived from Rodrigues' formula via Taylor approximation
    (in this case operating on the vector coefficients of the skew so(3) basis).

    ..math ::

        \left[\log(\mathbf{R})\right]^\lor = \frac{\theta}{2\sin(\theta)} \left[\mathbf{R} - \mathbf{R}^\top\right]^\lor

    This formula has problems at 1) angles theta close or equal to zero and 2) at angles close and
    equal to pi.

    To improve numerical stability for case 1), the angle term at small or zero angles is
    approximated by its truncated Taylor expansion:

    .. math ::

        \left[\log(\mathbf{R})\right]^\lor \approx \frac{1}{2} (1 + \frac{\theta^2}{6}) \left[\mathbf{R} - \mathbf{R}^\top\right]^\lor

    For angles close or equal to pi (case 2), the outer product relation can be used to obtain the
    squared rotation vector:

    .. math :: \omega \otimes \omega = \frac{1}{2}(\mathbf{I} + R)

    Taking the root of the diagonal elements recovers the normalized rotation vector up to the signs
    of the component. The latter can be obtained from the off-diagonal elements.

    Adapted from https://github.com/jasonkyuyim/se3_diffusion/blob/2cba9e09fdc58112126a0441493b42022c62bbea/data/so3_utils.py
    which was adapted from https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py
    with heavy help from https://cvg.cit.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf

    Args:
        rotation_matrices (torch.Tensor): Input batch of rotation matrices.

    Returns:
        torch.Tensor: Batch of rotation vectors.
    """
    # Get angles and sin/cos from rotation matrix.
    angles, angles_sin, _ = angle_from_rotmat(rotation_matrices)
    # Compute skew matrix representation and extract so(3) vector components.
    vector = skew_matrix_to_vector(rotation_matrices - rotation_matrices.transpose(-2, -1))

    # Three main cases for angle theta, which are captured
    # 1) Angle is 0 or close to zero -> use Taylor series for small values / return 0 vector.
    mask_zero = torch.isclose(angles, torch.zeros_like(angles)).to(angles.dtype)
    # 2) Angle is close to pi -> use outer product relation.
    mask_pi = torch.isclose(angles, torch.full_like(angles, np.pi), atol=1e-2).to(angles.dtype)
    # 3) Angle is unproblematic -> use the standard formula.
    mask_else = (1 - mask_zero) * (1 - mask_pi)

    # Compute case dependent pre-factor (1/2 for angle close to 0, angle otherwise).
    numerator = mask_zero / 2.0 + angles * mask_else
    # The Taylor expansion used here is actually the inverse of the Taylor expansion of the inverted
    # fraction sin(x) / x which gives better accuracy over a wider range (hence the minus and
    # position in denominator).
    denominator = (
        (1.0 - angles**2 / 6.0) * mask_zero  # Taylor expansion for small angles.
        + 2.0 * angles_sin * mask_else  # Standard formula.
        + mask_pi  # Avoid zero division at angle == pi.
    )
    prefactor = numerator / denominator
    vector = vector * prefactor[..., None]

    # For angles close to pi, derive vectors from their outer product (ww' = 1 + R).
    id3 = _broadcast_identity(rotation_matrices)
    skew_outer = (id3 + rotation_matrices) / 2.0
    # Ensure diagonal is >= 0 for square root (uses identity for masking).
    skew_outer = skew_outer + (torch.relu(skew_outer) - skew_outer) * id3

    # Get basic rotation vector as sqrt of diagonal (is unit vector).
    vector_pi = torch.sqrt(torch.diagonal(skew_outer, dim1=-2, dim2=-1))

    # Compute the signs of vector elements (up to a global phase).
    # Fist select indices for outer product slices with the largest norm.
    signs_line_idx = torch.argmax(torch.norm(skew_outer, dim=-1), dim=-1).long()
    # Select rows of outer product and determine signs.
    signs_line = torch.take_along_dim(skew_outer, dim=-2, indices=signs_line_idx[..., None, None])
    signs_line = signs_line.squeeze(-2)
    signs = torch.sign(signs_line)

    # Apply signs and rotation vector.
    vector_pi = vector_pi * angles[..., None] * signs

    # Fill entries for angle == pi in rotation vector (basic vector has zero entries at this point).
    vector = vector + vector_pi * mask_pi[..., None]

    return vector


def angle_from_rotmat(
    rotation_matrices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute rotation angles (as well as their sines and cosines) encoded by rotation matrices.
    Uses atan2 for better numerical stability for small angles.

    Args:
        rotation_matrices (torch.Tensor): Batch of rotation matrices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Batch of computed angles, sines of the
          angles and cosines of angles.
    """
    # Compute sine of angles (uses the relation that the unnormalized skew vector generated by a
    # rotation matrix has the length 2*sin(theta))
    skew_matrices = rotation_matrices - rotation_matrices.transpose(-2, -1)
    skew_vectors = skew_matrix_to_vector(skew_matrices)
    angles_sin = torch.norm(skew_vectors, dim=-1) / 2.0
    # Compute the cosine of the angle using the relation cos theta = 1/2 * (Tr[R] - 1)
    angles_cos = (torch.einsum("...ii", rotation_matrices) - 1.0) / 2.0

    # Compute angles using the more stable atan2
    angles = torch.atan2(angles_sin, angles_cos)

    return angles, angles_sin, angles_cos


def vector_to_skew_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """
    Map a vector into the corresponding skew matrix so(3) basis.
    ```
                [  0 -z  y]
    [x,y,z] ->  [  z  0 -x]
                [ -y  x  0]
    ```

    Args:
        vectors (torch.Tensor): Batch of vectors to be mapped to skew matrices.

    Returns:
        torch.Tensor: Vectors in skew matrix representation.
    """
    # Generate empty skew matrices.
    skew_matrices = torch.zeros((*vectors.shape, 3), device=vectors.device, dtype=vectors.dtype)

    # Populate positive values.
    skew_matrices[..., 2, 1] = vectors[..., 0]
    skew_matrices[..., 0, 2] = vectors[..., 1]
    skew_matrices[..., 1, 0] = vectors[..., 2]

    # Generate skew symmetry.
    skew_matrices = skew_matrices - skew_matrices.transpose(-2, -1)

    return skew_matrices


def skew_matrix_to_vector(skew_matrices: torch.Tensor) -> torch.Tensor:
    """
    Extract a rotation vector from the so(3) skew matrix basis.

    Args:
        skew_matrices (torch.Tensor): Skew matrices.

    Returns:
        torch.Tensor: Rotation vectors corresponding to skew matrices.
    """
    vectors = torch.zeros_like(skew_matrices[..., 0])
    vectors[..., 0] = skew_matrices[..., 2, 1]
    vectors[..., 1] = skew_matrices[..., 0, 2]
    vectors[..., 2] = skew_matrices[..., 1, 0]
    return vectors


def _rotquat_to_axis_angle(
    rotation_quaternions: torch.Tensor, tol: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Auxiliary routine for computing rotation angle and rotation axis from unit quaternions. To avoid
    complications, rotations vectors with angles below `tol` are set to zero.

    Args:
        rotation_quaternions (torch.Tensor): Rotation quaternions in [r, i, j, k] format.
        tol (float, optional): Threshold for small rotations. Defaults to 1e-7.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotation angles and axes.
    """
    # Compute rotation axis and normalize (accounting for small length axes).
    rotation_axes = rotation_quaternions[..., 1:]
    rotation_axes_norms = torch.norm(rotation_axes, dim=-1)

    # Compute rotation angle via atan2
    rotation_angles = 2.0 * torch.atan2(rotation_axes_norms, rotation_quaternions[..., 0])

    # Save division.
    rotation_axes = rotation_axes / (rotation_axes_norms[:, None] + tol)
    return rotation_angles, rotation_axes


def rotquat_to_rotvec(rotation_quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to rotation vectors.

    Args:
        rotation_quaternions (torch.Tensor): Input quaternions in [r,i,j,k] format.

    Returns:
        torch.Tensor: Rotation vectors.
    """
    rotation_angles, rotation_axes = _rotquat_to_axis_angle(rotation_quaternions)
    rotation_vectors = rotation_axes * rotation_angles[..., None]
    return rotation_vectors


def rotquat_to_rotmat(rotation_quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion to rotation matrix.

    Args:
        rotation_quaternions (torch.Tensor): Input quaternions in [r,i,j,k] format.

    Returns:
        torch.Tensor: Rotation matrices.
    """
    rotation_angles, rotation_axes = _rotquat_to_axis_angle(rotation_quaternions)
    skew_matrices = vector_to_skew_matrix(rotation_axes * rotation_angles[..., None])
    rotation_matrices = skew_matrix_exponential_map(rotation_angles, skew_matrices)
    return rotation_matrices


def apply_rotvec_to_rotmat(
    rotation_matrices: torch.Tensor,
    rotation_vectors: torch.Tensor,
    tol: float = 1e-7,
) -> torch.Tensor:
    """
    Update a rotation encoded in a rotation matrix with a rotation vector.

    Args:
        rotation_matrices: Input batch of rotation matrices.
        rotation_vectors: Input batch of rotation vectors.
        tol: Small offset for numerical stability.

    Returns:
        Updated rotation matrices.
    """
    # Convert vector to matrices.
    rmat_right = rotvec_to_rotmat(rotation_vectors, tol=tol)
    # Accumulate rotation.
    rmat_rotated = torch.einsum("...ij,...jk->...ik", rotation_matrices, rmat_right)
    return rmat_rotated


def rotmat_to_skew_matrix(mat: torch.Tensor) -> torch.Tensor:
    """
    Generates skew matrix for corresponding rotation matrix.

    Args:
        mat (torch.Tensor): Batch of rotation matrices.

    Returns:
        torch.Tensor: Skew matrices in the shapes of mat.
    """
    vec = rotmat_to_rotvec(mat)
    return vector_to_skew_matrix(vec)


def skew_matrix_to_rotmat(skew: torch.Tensor) -> torch.Tensor:
    """
    Generates rotation matrix for corresponding skew matrix.

    Args:
        skew (torch.Tensor): Batch of target 3 by 3 skew symmetric matrices.

    Returns:
        torch.Tensor: Rotation matrices in the shapes of skew.
    """
    vec = skew_matrix_to_vector(skew)
    return rotvec_to_rotmat(vec)


def hat(vector: torch.Tensor) -> torch.Tensor:
    """convert vector to so(3)"""
    return vector_to_skew_matrix(vector)

def vee(matrix: torch.Tensor) -> torch.Tensor:
    """convert so(3) to vector"""
    return skew_matrix_to_vector(matrix)  

def exp(matrix: torch.Tensor) -> torch.Tensor:
    """map so(3) to SO(3)"""
    return skew_matrix_to_rotmat(matrix)

def log(rotmat: torch.Tensor) -> torch.Tensor:
    """map SO(3) to so(3)"""
    return rotmat_to_skew_matrix(rotmat)

def rot_transpose(rotmat: torch.Tensor) -> torch.Tensor: 
    """rotation matrix inverse"""
    return rotmat.transpose(-2, -1)

def rot_mult(mat_1: torch.Tensor, mat_2: torch.Tensor) -> torch.Tensor:
    """Matrix multiply two rotation matrices with leading dimensions."""
    return torch.einsum("...ij,...jk->...ik", mat_1, mat_2)

def multidim_trace(mat: torch.Tensor) -> torch.Tensor:
    """Take the trace of a matrix with leading dimensions."""
    return torch.einsum("...ii->...", mat)

def geodesic_dist(mat_1: torch.Tensor, mat_2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the geodesic distance of two rotation matrices.

    Args:
        mat_1 (torch.Tensor): First rotation matrix.
        mat_2 (torch.Tensor): Second rotation matrix.

    Returns:
        Scalar for the geodesic distance between mat_1 and mat_2 with the same
        leading (i.e. batch) dimensions.
    """
    A = rotmat_to_skew_matrix(rot_mult(rot_transpose(mat_1), mat_2))
    return torch.sqrt(multidim_trace(rot_mult(A, rot_transpose(A))))

def expmap(tangent_vec: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
    """
    Map a point in the tangent space of base_point (SO(3)) to the manifold.

    Args:
        tangent_vec (torch.Tensor): Point in the tangent space of base_point.
        base_point (torch.Tensor): Point on the manifold in SO(3).

    Returns:
        torch.Tensor: Point on the manifold in SO(3).
    """
    lie_vec = rot_mult(rot_transpose(base_point), tangent_vec)
    return rot_mult(base_point, exp(lie_vec))

def logmap(point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
    """
    Logmap from base_point to point.

    Args:
        point (torch.Tensor): Point on the manifold in SO(3).
        base_point (torch.Tensor): Point on the manifold in SO(3).

    Returns:
        torch.Tensor: tangent_vec in the tangent space of base_point.
    """
    lie_point = rot_mult(rot_transpose(base_point), point)
    return rot_mult(base_point, log(lie_point))

DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4

def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.

    More specifically::

        bounds=(lower_bound, upper_bound)
        if lower_bound <= x <= upper_bound:
            acos_linear_extrapolation(x) = acos(x)
        elif x <= lower_bound: # 1st order Taylor approximation
            acos_linear_extrapolation(x)
                = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
        else:  # x >= upper_bound
            acos_linear_extrapolation(x)
                = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)

    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap

def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)

def so3_relative_angle(
    R1: torch.Tensor,
    R2: torch.Tensor,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = R1.double() @ R2.permute(0, 2, 1).double()
    return so3_rotation_angle(R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps)

def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)
        
def tangent_space_proj(base_point: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Project the given 3x3 matrix matrix onto the tangent space of SO(3) at base_point in PyTorch.
    
    Args:
    - matrix (torch.Tensor): a batch of 3x3 matrix from R^9
    - base_point (torch.Tensor): a batch of 3x3 matrix from SO(3) representing the point of tangency
    
    Returns:
    - T (torch.Tensor): projected 3x3 matrix in the tangent space of SO(3) at R
    """
    # Compute the skew-symmetric part of M
    skew_symmetric_part = 0.5 * (matrix - rot_transpose(matrix))
    
    # Project onto the tangent space at R
    return rot_mult(base_point, skew_symmetric_part)


def norm_SO3(base_point: torch.Tensor, tangent_vec: torch.Tensor) -> torch.Tensor:
    """calulate the norm squared of tangent_vec in the tangent space of base_point"""
    r = rot_mult(rot_transpose(base_point), tangent_vec)       # map backto so(3)
    norm = -torch.diagonal(r@r, dim1=-2, dim2=-1).sum(dim=-1)/2 #-trace(rTr)/2
    return norm

def norm_SO3_aa(base_point: torch.Tensor, tangent_vec: torch.Tensor) -> torch.Tensor:
    """calulate the norm squared of matrix T_R in the tangent space of R using axis-angle representation"""
    r = rot_mult(rot_transpose(base_point), tangent_vec)       # map backto so(3)
    r_aa = skew_matrix_to_vector(r)          # r_aa is the axis-angle representation of r, vector representation
    norm = torch.linalg.norm(r_aa, dim=-1)**2
    return norm

def geodesic_t(t: float, mat: torch.Tensor, base_mat: torch.Tensor, rot_vf=None) -> torch.Tensor:
    """
    Computes the geodesic at time t. Specifically, R_t = Exp_{base_mat}(t * Log_{base_mat}(mat)).

    Args:
        t: time along geodesic.
        mat: target points on manifold.
        base_mat: source point on manifold.

    Returns:
        Point along geodesic starting at base_mat and ending at mat.
    """
    if rot_vf is None:
        rot_vf = rotmat_to_rotvec(rot_mult(rot_transpose(base_mat), mat))
    # print(f"t:{t.shape},rot_vf:{rot_vf.shape}")
    # raise ValueError
    mat_t = rotvec_to_rotmat(t * rot_vf)
    if base_mat.shape != mat_t.shape:
        raise ValueError(
            f'Incompatible shapes: base_mat={base_mat.shape}, mat_t={mat_t.shape}')
    return torch.einsum("...ij,...jk->...ik", base_mat, mat_t)


def pairwise_geodesic_distance(x0, x1):
    """ Compute the pairwise geodisc distance between x0 and x1 on SO3.
        Parameters
        ----------
        x0 : Tensor, shape (bs, 3, 3)
            represents the source minibatch
        x1 : Tensor, shape (bs, 3, 3)
            represents the source minibatch

        Returns
        -------
        distances : Tensor, shape (bs, bs)
            represents the ground cost matrix between minibatches
    """
    batch_size = x0.size(0)
    x0 = rearrange(x0, 'b c d -> b (c d)', c=3, d=3)
    x1 = rearrange(x1, 'b c d -> b (c d)', c=3, d=3)
    mega_batch_x0 = rearrange(x0.repeat_interleave(batch_size, dim=0), 'b (c d) -> b c d', c=3, d=3)
    mega_batch_x1 = rearrange(x1.repeat(batch_size, 1), 'b (c d) -> b c d', c=3, d=3)
    distances = so3_relative_angle(mega_batch_x0, mega_batch_x1)**2
    return distances.reshape(batch_size, batch_size)



def calc_rot_vf(mat_t: torch.Tensor, mat_1: torch.Tensor) -> torch.Tensor:
    """
    Computes the vector field Log_{mat_t}(mat_1).

    Args:
        mat_t (torch.Tensor): base point to compute vector field at.
        mat_1 (torch.Tensor): target rotation.

    Returns:
        Rotation vector representing the vector field.
    """
    return rotmat_to_rotvec(rot_mult(rot_transpose(mat_t), mat_1))


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss

if __name__ == "__main__":
    #TODO: test rotation/vf loss
    pass
