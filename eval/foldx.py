import pandas as pd
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
import tempfile
import os
import shutil
import subprocess

from Bio.PDB import PDBParser

def fetch_stability_score(path):
    u = pd.read_csv(path, sep='\t', header=None)
    return u.values[0][1]

def fetch_binding_affinity(path):
    with open(path, 'r') as f:
        u = f.readlines()
    return float(u[-1].split("\t")[-3])
    
class FoldXSession(object):
    def __init__(self):
        super().__init__()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.pdb_names = []

    def cleanup(self):
        self.tmpdir.cleanup()
        self.tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    @property
    def workdir(self):
        return self.tmpdir.name

    def path(self, filename):
        return os.path.join(self.workdir, filename)

    def preprocess_data(self, pdb_dir, pdb_name):
        shutil.copy(os.path.join(pdb_dir, pdb_name), self.path(pdb_name))
        return self.path(pdb_name)

def get_chain_names(pdb_dir,pdb_name):
    pep_chain = pdb_name.split("_")[-1][0]
    parser = PDBParser()
    structure = parser.get_structure("name", os.path.join(pdb_dir,pdb_name))
    chain_names = [chain.get_id() for model in structure for chain in model]
    chains = f"{pep_chain},"
    for chain in chain_names:
        if chain != pep_chain:
            chains += f"{chain}"
    return chains

def process_one_file(pdb_dir,pdb_name):
    chains = get_chain_names(pdb_dir,pdb_name)
    with FoldXSession() as session:
        try:
            # print(session.workdir)
            session.preprocess_data(pdb_dir, pdb_name)
            assert(os.path.exists(session.path(pdb_name)))
            # print(os.listdir(session.workdir))
            ret = subprocess.run(['/datapool/data2/home/ruihan/bin/foldx', '--command='+'AnalyseComplex', '--pdb='+pdb_name, f'--analyseComplexChains={chains}'], cwd=session.workdir, stdout=None)
            fxout_path = session.path(f'Summary_{pdb_name.split(".")[0]}_AC.fxout')
            assert(os.path.exists(fxout_path))
            return (pdb_name.split('.')[0],fetch_binding_affinity(fxout_path))
        except:
            print(f"Error in {pdb_name}")
            print(os.path.exists(fxout_path))
            return (pdb_name.split('.')[0],None)
    