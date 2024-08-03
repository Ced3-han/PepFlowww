import subprocess
import re
from tqdm import tqdm
import os


RUNNER = '/datapool/data2/home/jiahan/Tool/TMalign-20180426/MMalign'

def align_pdb(pdb1,pdb2,pdb1_out):
    subprocess.run([RUNNER,pdb1,pdb2,'-o',pdb1_out],stdout=subprocess.PIPE)

def get_tm_score(pdb1,pdb2):
    cmd = subprocess.run(['TMscore',pdb1,pdb2],stdout=subprocess.PIPE)
    out = cmd.stdout.decode()
    tm_score = re.search(r"TM-score\s+=\s+(\d+\.\d+)", out)
    rmsd = re.search(r"RMSD of  the common residues=\s+(\d+\.\d+)", out)
    return float(rmsd.group(1)),float(tm_score.group(1))