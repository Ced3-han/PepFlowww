import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import math
from tqdm.auto import tqdm
import functools
import os
import argparse
import pandas as pd
from copy import deepcopy

from models_con.pep_dataloader import PepDataset

from pepflow.utils.train import recursive_to

from pepflow.modules.common.geometry import reconstruct_backbone, reconstruct_backbone_partially, align, batch_align
from pepflow.modules.protein.writers import save_pdb

from pepflow.utils.data import PaddingCollate

from models_con.utils import process_dic

from models_con.flow_model import FlowModel

from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask

collate_fn = PaddingCollate(eight=False)

import argparse


def item_to_batch(item, nums=32):
    data_list = [deepcopy(item) for i in range(nums)]
    return collate_fn(data_list)

def sample_for_data_bb(data, model, device, save_root, num_steps=200, sample_structure=True, sample_sequence=True, nums=8):
    if not os.path.exists(os.path.join(save_root,data["id"])):
        os.makedirs(os.path.join(save_root,data["id"]))
    batch = recursive_to(item_to_batch(data, nums=nums),device=device)
    traj = model.sample(batch, num_steps=num_steps, sample_structure=sample_structure, sample_sequence=sample_sequence)
    final = recursive_to(traj[-1], device=device)
    pos_bb = reconstruct_backbone(R=final['rotmats'],t=final['trans'],aa=final['seqs'],chain_nb=batch['chain_nb'],res_nb=batch['res_nb'],mask=batch['res_mask']) # (32,L,4,3)
    pos_ha = F.pad(pos_bb, pad=(0,0,0,15-4), value=0.) # (32,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
    mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
    mask_bb_atoms[:,:,:4] = True
    mask_new = torch.where(batch['generate_mask'][:,:,None],mask_bb_atoms,batch['mask_heavyatom'])
    aa_new = final['seqs']

    chain_nb = torch.LongTensor([0 if gen_mask else 1 for gen_mask in data['generate_mask']])
    chain_id = ['A' if gen_mask else 'B' for gen_mask in data['generate_mask']]
    icode = [' ' for _ in range(len(data['icode']))]
    for i in range(nums):
        ref_bb_pos = data['pos_heavyatom'][i][:,:4].cpu()
        pred_bb_pos = pos_new[i][:,:4].cpu()
        data_saved = {
                      'chain_nb':data['chain_nb'],'chain_id':data['chain_id'],'resseq':data['resseq'],'icode':data['icode'],
                      'aa':aa_new[i].cpu(), 'mask_heavyatom':mask_new[i].cpu(), 'pos_heavyatom':pos_new[i].cpu(),
                    }

        save_pdb(data_saved,path=os.path.join(save_root,data["id"],f'{data["id"]}_{i}.pdb'))
    save_pdb(data,path=os.path.join(save_root,data["id"],f'{data["id"]}_gt.pdb'))

def save_samples_bb(samples,save_dir):
    # meta data
    batch = recursive_to(samples['batch'],'cpu')
    chain_id = [list(item) for item in zip(*batch['chain_id'])][0] # fix chain id in collate func
    icode = [' ' for _ in range(len(chain_id))] # batch icode have same problem
    nums = len(batch['id'])
    id = batch['id'][0]
    # batch convert
    # aa=batch['aa] if only bb level
    pos_bb = reconstruct_backbone(R=samples['rotmats'],t=samples['trans'],aa=samples['seqs'],chain_nb=batch['chain_nb'],res_nb=batch['res_nb'],mask=batch['res_mask']) # (32,L,4,3)
    pos_ha = F.pad(pos_bb, pad=(0,0,0,15-4), value=0.) # (32,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
    mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
    mask_bb_atoms[:,:,:4] = True
    mask_new = torch.where(batch['generate_mask'][:,:,None],mask_bb_atoms,batch['mask_heavyatom'])
    aa_new = samples['seqs']
    for i in range(nums):
        data_saved = {
                      'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                      'aa':aa_new[i], 'mask_heavyatom':mask_new[i], 'pos_heavyatom':pos_new[i],
                    }
        save_pdb(data_saved,path=os.path.join(save_dir,f'sample_{i}.pdb'))
    data_saved = {
                    'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                    'aa':batch['aa'][0], 'mask_heavyatom':batch['mask_heavyatom'][0], 'pos_heavyatom':batch['pos_heavyatom'][0],
                }
    save_pdb(data_saved,path=os.path.join(save_dir,f'gt.pdb'))

def save_samples_sc(samples,save_dir):
    # meta data
    batch = recursive_to(samples['batch'],'cpu')
    chain_id = [list(item) for item in zip(*batch['chain_id'])][0] # fix chain id in collate func
    icode = [' ' for _ in range(len(chain_id))] # batch icode have same problem
    nums = len(batch['id'])
    id = batch['id'][0]
    # batch convert
    # aa=batch['aa] if only bb level
    pos_ha,_,_ = full_atom_reconstruction(R_bb=samples['rotmats'],t_bb=samples['trans'],angles=samples['angles'],aa=samples['seqs']) # (32,L,14,3), instead of 15, ignore OXT masked
    pos_ha = F.pad(pos_ha, pad=(0,0,0,15-14), value=0.) # (32,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
    mask_new = get_heavyatom_mask(samples['seqs'])
    aa_new = samples['seqs']
    for i in range(nums):
        data_saved = {
                      'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                      'aa':aa_new[i], 'mask_heavyatom':mask_new[i], 'pos_heavyatom':pos_new[i],
                    }
        save_pdb(data_saved,path=os.path.join(save_dir,f'sample_{i}.pdb'))
    data_saved = {
                    'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                    'aa':batch['aa'][0], 'mask_heavyatom':batch['mask_heavyatom'][0], 'pos_heavyatom':batch['pos_heavyatom'][0],
                }
    save_pdb(data_saved,path=os.path.join(save_dir,f'gt.pdb'))

if __name__ == '__main__':
    # sample = torch.load('./Codesign/outputs/1aze_B.pt')
    # save_samples_sc(sample,'./misc/test')
    # save_samples_bb(sample,'./misc/test')
    # for k,v in sample.items():
    #     if isinstance(v,torch.Tensor):
    #         print(f'{k},{v.shape}')

    # # subdir = 'bb_seq_angle' # bb,bb_seq,bb_seq_angle
    # names = [n.split('.')[0] for n in os.listdir(os.path.join(SAMPLE_DIR,subdir,'outputs'))]
    # for name in tqdm(names):
    #     sample = torch.load(os.path.join(SAMPLE_DIR,subdir,'outputs',f'{name}.pt'))
    #     os.makedirs(os.path.join(SAMPLE_DIR,subdir,'pdbs',name),exist_ok=True)
    #     save_samples_sc(sample,os.path.join(SAMPLE_DIR,subdir,'pdbs',name))
    
    args = argparse.ArgumentParser()
    args.add_argument('--SAMPLEDIR', type=str)
    parser = args.parse_args()
    SAMPLE_DIR = parser.SAMPLEDIR
    names = [n.split('.')[0] for n in os.listdir(os.path.join(SAMPLE_DIR,'outputs'))]
    for name in tqdm(names):
        sample = torch.load(os.path.join(SAMPLE_DIR,'outputs',f'{name}.pt'))
        os.makedirs(os.path.join(SAMPLE_DIR,'pdbs',name),exist_ok=True)
        save_samples_sc(sample,os.path.join(SAMPLE_DIR,'pdbs',name))