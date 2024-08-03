import torch
import math

from typing import Any, Optional, Union, cast

from pepflow.modules.common.geometry import *
import pepflow.modules.protein.constants as constants

"""
calc torsion angles between (0,2pi)
"""

def _get_torsion(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign( (torch.cross(v1, v2, dim=-1) * v0).sum(-1) )
    dihed = sgn*torch.acos( (n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999))
    return dihed

def get_chi_angles(restype, pos14):
    chi_angles = torch.full([4], fill_value=float("inf")).to(pos14)
    base_atom_names = constants.chi_angles_atoms[restype]
    for i, four_atom_names in enumerate(base_atom_names):
        atom_indices = [constants.restype_atom14_name_to_index[restype][a] for a in four_atom_names]
        p = torch.stack([pos14[i] for i in atom_indices])
        # if torch.eq(p, 99999).any():
        #     continue
        torsion = _get_torsion(*torch.unbind(p, dim=0))
        chi_angles[i] = torsion
    return chi_angles


def get_psi_angle(pos14: torch.Tensor) -> torch.Tensor:
    return _get_torsion(pos14[0], pos14[1], pos14[2], pos14[3]).reshape([1]) # af style psi, N,CA,C,O


def get_torsion_angle(pos14: torch.Tensor, aa: torch.LongTensor):
    torsion, torsion_mask = [], []
    for i in range(pos14.shape[0]):
        if aa[i] < constants.AA.UNK: # 0-19
            chi = get_chi_angles(aa[i].item(), pos14[i])
            psi = get_psi_angle(pos14[i])
            torsion_this = torch.cat([psi, chi], dim=0)
            torsion_mask_this = torsion_this.isfinite()
        else:
            torsion_this = torch.full([5], 0.)
            torsion_mask_this = torch.full([5], False)
        torsion.append(torsion_this.nan_to_num(posinf=0.))
        torsion_mask.append(torsion_mask_this)
    
    torsion = torch.stack(torsion) % (2*math.pi)
    torsion_mask = torch.stack(torsion_mask).bool()

    return torsion, torsion_mask

def _make_psi_chi_rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Compute psi and chi rotation matrices from torsional angles.

    Here we provide angles instead of alpha in af2 between (0,2pi)

    See alphafold supplementary Algorithm 25 for details.

    Args:
        angles: (B, N, 5), angles between (0,2pi)

    Returns:
        Torsional angle rotation matrices, (B, N, 5, 3, 3).
    """
    batch_size, n_res = angles.shape[:2]
    sine,cosine = torch.sin(angles), torch.cos(angles)
    sine = sine.reshape(batch_size, n_res, -1, 1, 1)
    cosine = cosine.reshape(batch_size, n_res, -1, 1, 1)
    zero = torch.zeros_like(sine)
    one = torch.ones_like(sine)

    row1 = torch.cat([one, zero, zero], dim=-1)  # (B, N, 5, 1, 3)
    row2 = torch.cat([zero, cosine, -sine], dim=-1)  # (B, N, 5, 1, 3)
    row3 = torch.cat([zero, sine, cosine], dim=-1)  # (B, N, 5, 1, 3)
    R = torch.cat([row1, row2, row3], dim=-2)  # (B, N, 5, 3, 3)

    return R


def _get_rigid_group(aa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract rigid group constants.

    Args:
        aa: Amino acid types, (B, N).

    Returns:
        A tuple of rigid group rotation, translation, atom14 group and atom14 position.
    """
    batch_size, n_res = aa.size()
    aa = aa.flatten()
    rotation = constants.restype_rigid_group_rotation.to(aa.device)[aa].reshape(batch_size, n_res, 8, 3, 3)
    translation = constants.restype_rigid_group_translation.to(aa.device)[aa].reshape(batch_size, n_res, 8, 3)
    atom14_group = constants.restype_heavyatom_to_rigid_group.to(aa.device)[aa].reshape(batch_size, n_res, 14)
    atom14_position = constants.restype_heavyatom_rigid_group_positions.to(aa.device)[aa].reshape(
        batch_size, n_res, 14, 3
    )
    return rotation, translation, atom14_group, atom14_position


# construct heavy atom masks for genrating
# restype_to_heavyatom_masks = {
#     restype: [name != "" and name !='OXT' for name in names]
#     for restype, names in constants.restype_to_heavyatom_names.items()
# }
# print(restype_to_heavyatom_masks[0])

restype_to_heavyatom_masks = torch.zeros([22,15]).bool()
for i in range(21):
    restype_to_heavyatom_masks[i] = torch.tensor([name != "" and name !='OXT' for name in constants.restype_to_heavyatom_names[i]]).bool()

def get_heavyatom_mask(aa: torch.Tensor) -> torch.Tensor:
    """Compute heavy atom masks from amino acid types.

    Args:
        aa: Amino acid types, (B, N).

    Returns:
        Heavy atom masks, (B, N, 15).
    """
    batch_size, n_res = aa.size()
    aa = aa.flatten()
    mask = restype_to_heavyatom_masks.to(aa.device)[aa].reshape(batch_size, n_res, 15)
    return mask

def full_atom_reconstruction(
    R_bb: torch.Tensor,
    t_bb: torch.Tensor,
    angles: torch.Tensor,
    aa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute full atom positions from backbone frames and torsional angles.

    See alphafold supplementary Algorithm 24 for details.

    Args:
        R_bb: Rotation of backbone frames, (B, N, 3, 3).
        t_bb: Translation of backbone frames, (B, N, 3).
        angles: (B, N, 5), angles between (0,2pi)
        aa: Amino acid types, (B, N).

    Returns:
        A tuple of atom positions and full frames, (pos14, R, t).
        pos14: Full atom positions in pos14 representations, (B, N, 14, 3).
        R: Rotation of backbone, psi, chi1-4 frames, (B, N, 5, 3, 3).
        t: Rotation of backbone, psi, chi1-4 frames, (B, N, 5, 3).
    """
    N, L = aa.size()

    rot_psi, rot_chi1, rot_chi2, rot_chi3, rot_chi4 = _make_psi_chi_rotation_matrices(angles).unbind(dim=2)
    # (B, N, 3, 3)
    zeros = torch.zeros_like(t_bb)

    rigid_rotation, rigid_translation, atom14_group, atom14_position = _get_rigid_group(aa)

    R_psi, t_psi = compose_chain(
        [
            (R_bb, t_bb),
            (rigid_rotation[:, :, constants.PSI_FRAME], rigid_translation[:, :, constants.PSI_FRAME]),
            (rot_psi, zeros),
        ]
    )

    R_chi1, t_chi1 = compose_chain(
        [
            (R_bb, t_bb),
            (rigid_rotation[:, :, constants.CHI1_FRAME], rigid_translation[:, :, constants.CHI1_FRAME]),
            (rot_chi1, zeros),
        ]
    )

    R_chi2, t_chi2 = compose_chain(
        [
            (R_chi1, t_chi1),
            (rigid_rotation[:, :, constants.CHI2_FRAME], rigid_translation[:, :, constants.CHI2_FRAME]),
            (rot_chi2, zeros),
        ]
    )

    R_chi3, t_chi3 = compose_chain(
        [
            (R_chi2, t_chi2),
            (rigid_rotation[:, :, constants.CHI3_FRAME], rigid_translation[:, :, constants.CHI3_FRAME]),
            (rot_chi3, zeros),
        ]
    )

    R_chi4, t_chi4 = compose_chain(
        [
            (R_chi3, t_chi3),
            (rigid_rotation[:, :, constants.CHI4_FRAME], rigid_translation[:, :, constants.CHI4_FRAME]),
            (rot_chi4, zeros),
        ]
    )

    # Return Frame
    R_ret = torch.stack([R_bb, R_psi, R_chi1, R_chi2, R_chi3, R_chi4], dim=2)
    t_ret = torch.stack([t_bb, t_psi, t_chi1, t_chi2, t_chi3, t_chi4], dim=2)

    # Backbone, Omega, Phi, Psi, Chi1,2,3,4
    R_all = torch.stack([R_bb, R_bb, R_bb, R_psi, R_chi1, R_chi2, R_chi3, R_chi4], dim=2)  # (B, N, 8, 3, 3)
    t_all = torch.stack([t_bb, t_bb, t_bb, t_psi, t_chi1, t_chi2, t_chi3, t_chi4], dim=2)  # (B, N, 8, 3)

    index_R = atom14_group.reshape(N, L, 14, 1, 1).repeat(1, 1, 1, 3, 3)  # (B, N, 14, 3, 3)
    index_t = atom14_group.reshape(N, L, 14, 1).repeat(1, 1, 1, 3)  # (B, N, 14, 3)

    R_atom = torch.gather(R_all, dim=2, index=index_R)  # (N, L, 14, 3, 3)
    t_atom = torch.gather(t_all, dim=2, index=index_t)  # (N, L, 14, 3)
    p_atom = atom14_position  # (N, L, 14, 3)

    pos14 = torch.matmul(R_atom, p_atom.unsqueeze(-1)).squeeze(-1) + t_atom
    return pos14, R_ret, t_ret



torsions_mask = torch.zeros([22,5]).float() # 0-19, X, PAD
for i in range(21):
    torsions_mask[i] = torch.tensor([True] + constants.chi_angles_mask[i]).float()
# print(angles_mask)

if __name__ =='__main__':
    aa = torch.full([3,8],fill_value=constants.AA.THR).long()
    mask = get_heavyatom_mask(aa)
    print(mask)
    print(mask.shape)