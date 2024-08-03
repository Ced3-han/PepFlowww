from utils import *

import os
import pandas as pd
import subprocess
import torch
import numpy as np
import shutil
from tqdm import tqdm

from joblib import delayed, Parallel

input_dir="./Baselines_new/Tests"
# output_dir="/datapool/data2/home/jiahan/Res Proj/PepDiff/frame-flow/Data/RF_samples"
output_dir="./Data/Baselines_new/Fixbb"

RUNNER = "/datapool/data2/home/jiahan/Tool/esm/examples/inverse_folding/sample_sequences.py"


def process_one_item_esmif(name='1a1m_C',chains_to_design="A",num_samples=10,temperature=0.1):
    if not os.path.exists(os.path.join(output_dir,name,'esms')):
        os.makedirs(os.path.join(output_dir,name,'esms'))
    assert os.path.exists(os.path.join(output_dir,name,'esms'))
    # if not os.path.exists(os.path.join(output_dir,name,'pocket_merge_renum.pdb')):
    #     chain_dic = renumber_pdb(os.path.join(input_dir,name,'pocket_merge.pdb'),os.path.join(output_dir,name,'pocket_merge_renum.pdb'))
    dirname = os.path.join(output_dir,name,'esms')
    cmd = [
    "python", RUNNER, os.path.join(input_dir,name,'pocket_merge_renum.pdb'),
    "--chain", chains_to_design, "--temperature", f"{temperature}", "--num-samples", f"{num_samples}",
    "--outpath", os.path.join(dirname,'pocket_merge_renum.fasta'),
    "--multichain-backbone", "--nogpu"
]
    subprocess.run(cmd)
