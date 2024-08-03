from utils import *
from geometry import *

import os
import pandas as pd
import subprocess
import torch
import numpy as np
import shutil
from tqdm import tqdm

from joblib import delayed, Parallel

from Bio.PDB import PDBParser, PDBIO, Select


HELPERS = "/datapool/data2/home/jiahan/Tool/ProteinMPNN/helper_scripts"
RUNNER = "/datapool/data2/home/jiahan/Tool/ProteinMPNN/protein_mpnn_run.py"

def get_chain_nums(pdb_path,chain_id):
    parser = PDBParser()
    chain = parser.get_structure('X',pdb_path)[0][chain_id]
    residue_nums = [residue.get_id()[1] for residue in chain]
    return residue_nums

def process_mpnn_bb(name='1aze_B',chains_to_design="A",num_samples=1):
    input_dir = './Data/Models_new/Codesign/bb/pdbs'
    output_dir = './Data/Models_new/Codesign/bb/seqs'
    if not os.path.exists(os.path.join(output_dir,name)):
        os.makedirs(os.path.join(output_dir,name))
    dirname = os.path.join(output_dir,name)
    # defined dirs
    path_for_parsed_chains=os.path.join(dirname,'parsed_pdbs.jsonl')
    path_for_assigned_chains=os.path.join(dirname,'assigned_pdbs.jsonl')
    path_for_fixed_positions=os.path.join(dirname,'fixed_pdbs.jsonl')
    residue_nums = get_chain_nums(os.path.join(input_dir,name,'gt.pdb'),chains_to_design)
    design_only_positions = " ".join(map(str,residue_nums)) #design only these residues; use flag --specify_non_fixed
    # print(path_for_assigned_chains)
    # print(design_only_positions)
    subprocess.run([
        "python", os.path.join(HELPERS,"parse_multiple_chains.py"),
        "--input_path", os.path.join(input_dir,name),
        "--output_path", path_for_parsed_chains,
    ])
    subprocess.run([
        "python", os.path.join(HELPERS,"assign_fixed_chains.py"),
        "--input_path", path_for_parsed_chains,
        "--output_path", path_for_assigned_chains,
        '--chain_list', chains_to_design,
    ])
    subprocess.run([
        "python", os.path.join(HELPERS,"make_fixed_positions_dict.py"),
        "--input_path", path_for_parsed_chains,
        "--output_path", path_for_fixed_positions,
        '--chain_list', chains_to_design,
        '--position_list', design_only_positions,
        '--specify_non_fixed'
    ])
    # run mpnn
    # print('run mpnns')
    subprocess.run([
        "python", RUNNER,
        "--jsonl_path", path_for_parsed_chains,
        "--chain_id_jsonl", path_for_assigned_chains,
        "--fixed_positions_jsonl", path_for_fixed_positions,
        "--out_folder", dirname,
        "--num_seq_per_target", f"{num_samples}",
        "--sampling_temp", "0.1",
        "--seed", "37",
        "--batch_size","1",
        '--device','cuda:1'
    ])

def process_one_item_mpnn(name='1a1m_C',chains_to_design="A",num_samples=1):
    input_dir="./Data/Baselines_new/Tests"
    output_dir="./Data/Baselines_new/Codesign"
    if not os.path.exists(os.path.join(output_dir,name,'mpnns')):
        os.makedirs(os.path.join(output_dir,name,'mpnns'))
    # if not os.path.exists(os.path.join(output_dir,name,'pocket_merge_renum.pdb')):
    #     chain_dic = renumber_pdb(os.path.join(input_dir,name,'pocket_merge.pdb'),os.path.join(output_dir,name,'pocket_merge_renum.pdb'))
    dirname = os.path.join(output_dir,name,'mpnns')
    # defined dirs
    path_for_parsed_chains=os.path.join(dirname,'parsed_pdbs.jsonl')
    path_for_assigned_chains=os.path.join(dirname,'assigned_pdbs.jsonl')
    path_for_fixed_positions=os.path.join(dirname,'fixed_pdbs.jsonl')
    with open(os.path.join(input_dir,name,'seq.fasta'),'r') as f:
        pep_len = len(f.readlines()[1].strip())
    design_only_positions=" ".join(map(str,list(range(1,pep_len+1)))) #design only these residues; use flag --specify_non_fixed
    # print(design_only_positions)
    # parsed chains
    # print("parsing chains")
    subprocess.run([
        "python", os.path.join(HELPERS,"parse_multiple_chains.py"),
        "--input_path", os.path.join('./Data/Baselines_new/Codesign',name,'rfs'),#os.path.join('/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/Data/Baselines/Fixbb/',name),
        "--output_path", path_for_parsed_chains,
    ])
    subprocess.run([
        "python", os.path.join(HELPERS,"assign_fixed_chains.py"),
        "--input_path", path_for_parsed_chains,
        "--output_path", path_for_assigned_chains,
        '--chain_list', chains_to_design,
    ])
    subprocess.run([
        "python", os.path.join(HELPERS,"make_fixed_positions_dict.py"),
        "--input_path", path_for_parsed_chains,
        "--output_path", path_for_fixed_positions,
        '--chain_list', chains_to_design,
        '--position_list', design_only_positions,
        '--specify_non_fixed'
    ])
    # run mpnn
    # print('run mpnns')
    subprocess.run([
        "python", RUNNER,
        "--jsonl_path", path_for_parsed_chains,
        "--chain_id_jsonl", path_for_assigned_chains,
        "--fixed_positions_jsonl", path_for_fixed_positions,
        "--out_folder", dirname,
        "--num_seq_per_target", f"{num_samples}",
        "--sampling_temp", "0.1",
        "--seed", "37",
        "--batch_size","1",
        '--device','cuda:1'
    ])


def write_seq_to_pdb(seq_path,pdb_path,out_path,chain_id):
    # first we should fix GGGGG in rfs with mpnn generated seq
    aa_mapping = {"A": "ALA","C": "CYS","D": "ASP","E": "GLU","F": "PHE","G": "GLY","H": "HIS","I": "ILE","K": "LYS","L": "LEU","M": "MET","N": "ASN","P": "PRO","Q": "GLN","R": "ARG","S": "SER","T": "THR","V": "VAL","W": "TRP","Y": "TYR",
                  'X':'UNK'}
    tmps = []
    for record in SeqIO.parse(seq_path, "fasta"):
        tmps.append(str(record.seq))
    seq = tmps[-1]
    
    parser = PDBParser()
    structure = parser.get_structure("X", pdb_path)
    model = structure[0]
    for chain in model:
        if chain.id == chain_id:  # 假设你要更改的是链A
            for i,res in enumerate(chain):
                if i<len(seq):
                    res.resname = aa_mapping[seq[i]]
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)