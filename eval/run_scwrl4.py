from utils import *

import os
import pandas as pd
import subprocess
import numpy as np
import shutil
from tqdm import tqdm

from joblib import delayed, Parallel

input_dir="./Data/Baselines_new/Tests"
output_dir="./Data/Baselines_new/Pack"

RUNNER = "/datapool/data2/home/jiahan/Tool/bin/Scwrl4"

def process_one_item_scwrl4(name='1a1m_C',num_samples=10):
    if not os.path.exists(os.path.join(output_dir,name,'scwrls')):
        os.makedirs(os.path.join(output_dir,name,'scwrls'))
    # if not os.path.exists(os.path.join(output_dir,name,'pocket_merge_renum.pdb')):
    #     chain_dic = renumber_pdb(os.path.join(input_dir,name,'pocket_merge.pdb'),os.path.join(output_dir,name,'pocket_merge_renum.pdb'))
    #     keep_backbone_atoms(os.path.join(output_dir,name,'pocket_merge_renum.pdb'),os.path.join(output_dir,name,'pocket_merge_renum_backbone.pdb'))
    dirname = os.path.join(output_dir,name,'scwrls')
    for i in range(num_samples):
        cmd = [
            RUNNER,
            '-i',os.path.join(input_dir,name,'pocket_merge_renum_bb.pdb'),
            '-o',os.path.join(dirname,f'packed_{i}.pdb'),
        ]
        subprocess.run(cmd)