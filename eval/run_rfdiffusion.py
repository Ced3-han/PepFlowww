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

input_dir="./Data/Baselines_new/Tests"
output_dir=".Data/Baselines_new/Codesign"

PROGEN="/datapool/data2/home/jiahan/Tool/protein_generator/inference.py"

def process_one_item_rf(name='1a1m_C',num_samples=10):
    if not os.path.exists(os.path.join(output_dir,name,'rfs')):
        os.makedirs(os.path.join(output_dir,name,'rfs'))
    chain_dic = get_chain_dic(os.path.join(input_dir,name,'pocket_renum.pdb'))
    with open(os.path.join(input_dir,name,'seq.fasta'),'r') as f:
        pep_len = len(f.readlines()[1].strip())
    # rfdiffusion
    contigs = []
    for chain,chain_len in chain_dic.items():
        contigs.append(f'{chain}1-{chain_len}/0')
    contigs.append(f'{pep_len}-{pep_len}')
    contigs = " ".join(contigs)
    command = [
    "run_inference.py",
    f"inference.output_prefix='{os.path.join(output_dir,name,'rfs','sample')}'",
    f"inference.input_pdb='{os.path.join(input_dir,name,'pocket_renum.pdb')}'",
    f"contigmap.contigs=[{contigs}]",
    f"inference.num_designs={num_samples}",
]
    # print(command)
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return name
    except:
        return None
    
def process_one_item_pg(name='1a1m_C',num_samples=10):
    if not os.path.exists(os.path.join(output_dir,name,'pgs')):
        os.makedirs(os.path.join(output_dir,name,'pgs'))
    os.makedirs(os.path.join(output_dir,name,'pgs'),exist_ok=True)
    chain_dic = get_chain_dic(os.path.join(input_dir,name,'pocket_renum.pdb'))
    with open(os.path.join(input_dir,name,'seq.fasta'),'r') as f:
        pep_len = len(f.readlines()[1].strip())
    # protein_generator settings
    contigs = []
    for chain,chain_len in chain_dic.items():
        contigs.append(f'{chain}1-{chain_len},0')
    contigs.append(f'{pep_len}-{pep_len}')
    command = [
        "python", PROGEN,
        "--num_designs", f"{num_samples}",
        "--out", os.path.join(output_dir,name,'pgs','sample'),
        "--pdb", os.path.join(input_dir,name,'pocket_renum.pdb'),
        "--T", "25", # default setting
        "--save_best_plddt", # default setting
        "--contigs", *contigs,
    ]
    # print(command)
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return name
    except:
        return None

def process_one_item(name='1a1m_C',num_samples=10):
    process_one_item_pg(name,num_samples)
    process_one_item_rf(name,num_samples)