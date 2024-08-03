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

from pepflow.utils.misc import load_config
from pepflow.utils.train import recursive_to

from pepflow.modules.common.geometry import reconstruct_backbone, reconstruct_backbone_partially, align, batch_align
from pepflow.modules.protein.writers import save_pdb

from pepflow.utils.data import PaddingCollate

from models_con.utils import process_dic

import gc

from models_con.flow_model import FlowModel

from pepflow.utils.misc import seed_all

from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask

collate_fn = PaddingCollate(eight=False)

import argparse


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str)
    args.add_argument('--device', type=str)
    args.add_argument('--ckpt', type=str)
    args.add_argument('--output', type=str)
    args.add_argument('--num_steps', type=int, default=200)
    args.add_argument('--num_samples', type=int, default=64)
    args.add_argument('--sample_bb', type=bool, default=True)
    args.add_argument('--sample_ang', type=bool, default=True)
    args.add_argument('--sample_seq', type=bool, default=True)
    args.add_argument('--num_samples', type=int, default=64)
    args.add_argument('--num_samples', type=int, default=64)
    parser = args.parse_args()

    config,cfg_name = load_config(parser.config)
    device = parser.device
    dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
                                            name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate(eight=False), num_workers=4, pin_memory=True)
    ckpt = torch.load(parser.ckpt, map_location=device)

    seed_all(114514)
    model = FlowModel(config.model).to(device)
    model.load_state_dict(process_dic(ckpt['model']))
    model.eval()


    dic = {'id':[],'len':[],'tran':[],'aar':[],'rot':[],'trans_loss':[],'rot_loss':[]}

    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        data_list = [deepcopy(item) for _ in range(parser.num_samples)]
        batch = recursive_to(collate_fn(data_list),device)
        loss_dic = model(batch)
        traj_1 = model.sample(batch,num_steps=parser.num_steps,sample_bb=parser.sample_bb,sample_ang=parser.sample_ang,sample_seq=parser.sample_seq)
        ca_dist = torch.sqrt(torch.sum((traj_1[-1]['trans']-traj_1[-1]['trans_1'])**2*batch['generate_mask'][...,None].cpu().long()) / (torch.sum(batch['generate_mask']) + 1e-8).cpu()) # rmsd
        rot_dist = torch.sqrt(torch.sum((traj_1[-1]['rotmats']-traj_1[-1]['rotmats_1'])**2*batch['generate_mask'][...,None,None].long().cpu()) / (torch.sum(batch['generate_mask']) + 1e-8).cpu()) # rmsd
        aar = torch.sum((traj_1[-1]['seqs']==traj_1[-1]['seqs_1']) * batch['generate_mask'].long().cpu()) / (torch.sum(batch['generate_mask']).cpu() + 1e-8)
        

        print(loss_dic)
        print(f'tran:{ca_dist},rot:{rot_dist},aar:{aar},len:{batch["generate_mask"].sum().item()}')

        # free
        torch.cuda.empty_cache()
        gc.collect()
        
        dic['tran'].append(ca_dist.item())
        dic['rot'].append(rot_dist.item())
        dic['aar'].append(aar.item())
        dic['trans_loss'].append(loss_dic['trans_loss'].item())
        dic['rot_loss'].append(loss_dic['rot_loss'].item())
        dic['id'].append(batch['id'][0])
        dic['len'].append(batch['generate_mask'].sum().item())
        # break

        traj_1[-1]['batch'] = batch
        torch.save(traj_1[-1],f'{parser.output}/outputs/{batch["id"][0]}.pt')
    dic = pd.DataFrame(dic)
    dic.to_csv(f'{parser.output}/outputs.csv',index=None)