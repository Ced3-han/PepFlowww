import os
import pandas as pd
import subprocess
import torch
import esm
import numpy as np
import shutil
from tqdm import tqdm

from joblib import delayed, Parallel

import warnings
from Bio import BiopythonWarning, SeqIO

from geometry import *

# 忽略PDBConstructionWarning
warnings.filterwarnings('ignore', category=BiopythonWarning)

input_dir="./Data/Baselines_new/Tests"
output_dir="/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/Data/Baselines_new/Codesign"

model = esm.pretrained.esmfold_v1()
model = model.eval().to('cuda:2')

def process_rf(name='1aze_B'):
    input_dir=".Data/Baselines_new/Tests"
    output_dir=".Data/Baselines_new/Codesign"
    struct_dir = os.path.join(output_dir,name,'rfs_refold')
    seq_dir = os.path.join(output_dir,name,'mpnns','seqs')
    os.makedirs(struct_dir,exist_ok=True)
    seqs = {}
    for seq_path in os.listdir(seq_dir):
        tmp_seqs = []
        if seq_path.endswith('.fasta'):
            for record in SeqIO.parse(os.path.join(seq_dir,seq_path), "fasta"):
                tmp_seqs.append(str(record.seq))
        seqs[seq_path.split('.')[0]] = tmp_seqs[-1]
    for seq_name,seq in seqs.items():
        with torch.no_grad():
            output = model.infer_pdb(seq)
        with open(os.path.join(struct_dir,seq_name+'.pdb'),'w') as f:
            f.write(output)

def process_pg(name='1aze_B',chain_id='A'):
    input_dir=".Data/Baselines_new/Tests"
    output_dir=".Data/Baselines_new/Codesign"
    struct_dir = os.path.join(output_dir,name,'pgs_refold')
    seq_dir = os.path.join(output_dir,name,'pgs')
    os.makedirs(struct_dir,exist_ok=True)
    seqs = {}
    for seq_path in os.listdir(seq_dir):
        if seq_path.endswith('.pdb'):
            seqs[seq_path.split('.')[0]] = get_seq(os.path.join(seq_dir,seq_path),chain_id)
    for seq_name,seq in seqs.items():
        with torch.no_grad():
            output = model.infer_pdb(seq)
        with open(os.path.join(struct_dir,seq_name+'.pdb'),'w') as f:
            f.write(output)

def refold(name,chain_id,sub_dir):
    raw_dir = os.path.join('/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/Data/Models_new/Codesign',sub_dir,'pdbs')
    refold_dir = os.path.join('/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/Data/Models_new/Codesign',sub_dir,'pdbs_refold')
    os.makedirs(os.path.join(refold_dir,name),exist_ok=True)
    seqs = {}
    for seq_path in os.listdir(os.path.join(raw_dir,name)):
        if seq_path.endswith('.pdb'):
            seqs[seq_path.split('.')[0]] = get_seq(os.path.join(raw_dir,name,seq_path),chain_id)
    for seq_name,seq in seqs.items():
        with torch.no_grad():
            output = model.infer_pdb(seq)
        with open(os.path.join(refold_dir,name,seq_name+'.pdb'),'w') as f:
            f.write(output)