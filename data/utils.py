from typing import List, Dict, Any
from openfold.utils import rigid_utils as ru
from data import residue_constants
import numpy as np
import collections
import string
import pickle
import os
import torch
from torch_scatter import scatter_add, scatter
from Bio.PDB.Chain import Chain
from data import protein
import dataclasses
from Bio import PDB
import io

"""From https://github.com/microsoft/protein-frame-flow"""

Rigid = ru.Rigid
Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]

to_numpy = lambda x: x.detach().cpu().numpy()
aatype_to_seq = lambda aatype: ''.join([
        residue_constants.restypes_with_x[x] for x in aatype])

def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)


def batch_align_structures(pos_1, pos_2, mask=None):
    if pos_1.shape != pos_2.shape:
        raise ValueError('pos_1 and pos_2 must have the same shape.')
    if pos_1.ndim != 3:
        raise ValueError(f'Expected inputs to have shape [B, N, 3]')
    num_batch = pos_1.shape[0]
    device = pos_1.device
    batch_indices = (
        torch.ones(*pos_1.shape[:2], device=device, dtype=torch.int64) 
        * torch.arange(num_batch, device=device)[:, None]
    )
    flat_pos_1 = pos_1.reshape(-1, 3)
    flat_pos_2 = pos_2.reshape(-1, 3)
    flat_batch_indices = batch_indices.reshape(-1)
    if mask is None:
        aligned_pos_1, aligned_pos_2, align_rots = align_structures(
            flat_pos_1, flat_batch_indices, flat_pos_2)
        aligned_pos_1 = aligned_pos_1.reshape(num_batch, -1, 3)
        aligned_pos_2 = aligned_pos_2.reshape(num_batch, -1, 3)
        return aligned_pos_1, aligned_pos_2, align_rots

    flat_mask = mask.reshape(-1).bool()
    _, _, align_rots = align_structures(
        flat_pos_1[flat_mask],
        flat_batch_indices[flat_mask],
        flat_pos_2[flat_mask]
    )
    aligned_pos_1 = torch.bmm(
        pos_1,
        align_rots
    )
    return aligned_pos_1, pos_2, align_rots

