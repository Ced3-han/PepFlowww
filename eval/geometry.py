from Bio.PDB import PDBParser, Superimposer, is_aa, Select, NeighborSearch
import tmtools
import os
import numpy as np
import mdtraj as md
from Bio.SeqUtils import seq1

import warnings
from Bio import BiopythonWarning, SeqIO

import difflib
import torch

# 忽略PDBConstructionWarning
warnings.filterwarnings('ignore', category=BiopythonWarning)

def get_chain_from_pdb(pdb_path, chain_id='A'):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb_path)[0]
    for chain in structure:
        if chain.id == chain_id:
            # print(len(chain))
            return chain
    return None

def diff_ratio(str1, str2):
    # Create a SequenceMatcher object
    seq_matcher = difflib.SequenceMatcher(None, str1, str2)

    # Calculate the difference ratio
    return seq_matcher.ratio()

#######################################

#RMSD and Tm

#######################################
def align_chains(chain1, chain2):
    reslist1 = []
    reslist2 = []
    for residue1,residue2 in zip(chain1.get_residues(),chain2.get_residues()):
        if is_aa(residue1) and residue1.has_id('CA'): # at least have CA
            reslist1.append(residue1)
            reslist2.append(residue2)
    return reslist1,reslist2

def get_rmsd(chain1, chain2):
    # chain1 = get_chain_from_pdb(pdb1, chain_id1)
    # chain2 = get_chain_from_pdb(pdb2, chain_id2)
    if chain1 is None or chain2 is None:
        return None
    super_imposer = Superimposer()
    pos1 = np.array([atom.get_coord() for atom in chain1.get_atoms() if atom.name == 'CA'])
    pos2 = np.array([atom.get_coord() for atom in chain2.get_atoms() if atom.name == 'CA'])
    rmsd1 = np.sqrt(np.sum((pos1 - pos2)**2) / len(pos1))
    super_imposer.set_atoms([atom for atom in chain1.get_atoms() if atom.name == 'CA'],
                            [atom for atom in chain2.get_atoms() if atom.name == 'CA'])
    rmsd2 = super_imposer.rms
    return rmsd1,rmsd2

def get_tm(chain1,chain2):
    # chain1 = get_chain_from_pdb(pdb1, chain_id1)
    # chain2 = get_chain_from_pdb(pdb2, chain_id2)
    pos1 = np.array([atom.get_coord() for atom in chain1.get_atoms() if atom.name == 'CA'])
    pos2 = np.array([atom.get_coord() for atom in chain2.get_atoms() if atom.name == 'CA'])
    tm_results = tmtools.tm_align(pos1, pos2, 'A'*len(pos1), 'A'*len(pos2))
    # print(dir(tm_results))
    return tm_results.tm_norm_chain2

def get_traj_chain(pdb, chain):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    chain2id = {chain.id:i for i,chain in enumerate(structure)}
    traj = md.load(pdb)
    chain_indices = traj.topology.select(f"chainid {chain2id[chain]}")
    traj = traj.atom_slice(chain_indices)
    return traj

def get_second_stru(pdb,chain):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    chain2id = {chain.id:i for i,chain in enumerate(structure)}
    traj = md.load(pdb)
    chain_indices = traj.topology.select(f"chainid {chain2id[chain]}")
    traj = traj.atom_slice(chain_indices)
    return md.compute_dssp(traj,simplified=True)

def get_ss(traj1,traj2):
    # traj1,traj2 = get_traj_chain(pdb1,chain_id1),get_traj_chain(pdb2,chain_id2)
    ss1,ss2 = md.compute_dssp(traj1,simplified=True),md.compute_dssp(traj2,simplified=True)
    return (ss1==ss2).mean()

def get_bind_site(pdb,chain_id):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    peps = [atom for res in structure[chain_id] for atom in res if atom.get_name() == 'CA']
    recs = [atom for chain in structure if chain.get_id()!=chain_id for res in chain for atom in res if atom.get_name() == 'CA']
    # print(recs)
    search = NeighborSearch(recs)
    near_res = []
    for atom in peps:
        near_res += search.search(atom.get_coord(), 10.0, level='R')
    near_res = set([res.get_id()[1] for res in near_res])
    return near_res

def get_bind_ratio(pdb1, pdb2, chain_id1, chain_id2):
    near_res1,near_res2 = get_bind_site(pdb1,chain_id1),get_bind_site(pdb2,chain_id2)
    # print(near_res1)
    # print(near_res2)
    return len(near_res1.intersection(near_res2))/(len(near_res2)+1e-10) # last one is gt

def get_dihedral(pdb,chain):
    traj = get_traj_chain(pdb,chain)
    #TODO: dihedral

def get_seq(pdb,chain_id):
    parser = PDBParser()
    chain = parser.get_structure('X', pdb)[0][chain_id]
    return seq1("".join([residue.get_resname() for residue in chain])) # ignore is_aa,used for extract seq from genrated pdb

def get_mpnn_seqs(path):
    fastas = []
    for record in SeqIO.parse(path, "fasta"):
        tmp = [c for c in str(record.seq)]
        fastas.append(tmp)
    return fastas

