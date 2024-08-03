import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
from tqdm.auto import tqdm
import functools
from torch.utils.data import DataLoader
import os
import argparse

import pandas as pd

from models_con.edge import EdgeEmbedder
from models_con.node import NodeEmbedder
from pepflow.modules.common.layers import sample_from, clampped_one_hot
from models_con.ga import GAEncoder
from pepflow.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms
from pepflow.modules.common.geometry import construct_3d_basis
from pepflow.utils.data import mask_select_data, find_longest_true_segment, PaddingCollate
from pepflow.utils.misc import seed_all
from pepflow.utils.train import sum_weighted_losses
from torch.nn.utils import clip_grad_norm_

from pepflow.modules.so3.dist import centered_gaussian,uniform_so3
from pepflow.modules.common.geometry import batch_align, align

from tqdm import tqdm

import wandb

from data import so3_utils
from data import all_atom

from models_con.pep_dataloader import PepDataset

from pepflow.utils.misc import load_config
from pepflow.utils.train import recursive_to
from easydict import EasyDict

from models_con.utils import process_dic
from models_con.torsion import get_torsion_angle, torsions_mask
import models_con.torus as torus

import gc

from copy import deepcopy
from pepflow.utils.data import PaddingCollate
collate_fn = PaddingCollate(eight=False)
from pepflow.utils.train import recursive_to

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}

class FlowModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._model_cfg = cfg.encoder
        self._interpolant_cfg = cfg.interpolant

        self.node_embedder = NodeEmbedder(cfg.encoder.node_embed_size,max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(cfg.encoder.edge_embed_size,max_num_heavyatoms)
        self.ga_encoder = GAEncoder(cfg.encoder.ipa)

        self.sample_structure = self._interpolant_cfg.sample_structure
        self.sample_sequence = self._interpolant_cfg.sample_sequence

        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value
    
    def encode(self, batch):
        rotmats_1 =  construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],batch['pos_heavyatom'][:, :, BBHeavyAtom.C],batch['pos_heavyatom'][:, :, BBHeavyAtom.N] )
        trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        seqs_1 = batch['aa']

        # ignore psi
        # batch['torsion_angle'] = batch['torsion_angle'][:,:,1:]
        # batch['torsion_angle_mask'] = batch['torsion_angle_mask'][:,:,1:]
        angles_1 = batch['torsion_angle']

        context_mask = torch.logical_and(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
        structure_mask = context_mask if self.sample_structure else None
        sequence_mask = context_mask if self.sample_sequence else None
        node_embed = self.node_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)
        edge_embed = self.edge_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)
        
        return rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed
    
    def zero_center_part(self,pos,gen_mask,res_mask):
        """
        move pos by center of gen_mask
        pos: (B,N,3)
        gen_mask, res_mask: (B,N)
        """
        center = torch.sum(pos * gen_mask[...,None], dim=1) / (torch.sum(gen_mask,dim=-1,keepdim=True) + 1e-8) # (B,N,3)*(B,N,1)->(B,3)/(B,1)->(B,3)
        center = center.unsqueeze(1) # (B,1,3)
        # center = 0. it seems not center didnt influence the result, but its good for training stabilty
        pos = pos - center
        pos = pos * res_mask[...,None]
        return pos,center
    
    def seq_to_simplex(self,seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k # (B,L,K)
    
    def forward(self, batch):

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask,angle_mask = batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()

        #encode
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch) # no generate mask

        # prepare for denoise
        trans_1_c,_ = self.zero_center_part(trans_1,gen_mask,res_mask)
        trans_1_c = trans_1 # already centered when constructing dataset
        seqs_1_simplex = self.seq_to_simplex(seqs_1)
        seqs_1_prob = F.softmax(seqs_1_simplex,dim=-1)

        with torch.no_grad():
            t = torch.rand((num_batch,1), device=batch['aa'].device) 
            t = t*(1-2 * self._interpolant_cfg.t_normalization_clip) + self._interpolant_cfg.t_normalization_clip # avoid 0
            if self.sample_structure:
                # corrupt trans
                trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device) * self._interpolant_cfg.trans.sigma # scale with sigma?
                trans_0_c,_ = self.zero_center_part(trans_0,gen_mask,res_mask)
                trans_t = (1-t[...,None])*trans_0_c + t[...,None]*trans_1_c
                trans_t_c = torch.where(batch['generate_mask'][...,None],trans_t,trans_1_c)
                # corrupt rotmats
                rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
                rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
                rotmats_t = torch.where(batch['generate_mask'][...,None,None],rotmats_t,rotmats_1)
                # corrup angles
                angles_0 = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype) # (B,L,5)
                angles_t = torus.tor_geodesic_t(t[..., None], angles_1, angles_0)
                angles_t = torch.where(batch['generate_mask'][...,None],angles_t,angles_1)
            else:
                trans_t_c = trans_1_c.detach().clone()
                rotmats_t = rotmats_1.detach().clone()
                angles_t = angles_1.detach().clone()
            if self.sample_sequence:
                # corrupt seqs
                seqs_0_simplex = self.k * torch.randn_like(seqs_1_simplex) # (B,L,K)
                seqs_0_prob = F.softmax(seqs_0_simplex,dim=-1) # (B,L,K)
                seqs_t_simplex = ((1 - t[..., None]) * seqs_0_simplex) + (t[..., None] * seqs_1_simplex) # (B,L,K)
                seqs_t_simplex = torch.where(batch['generate_mask'][...,None],seqs_t_simplex,seqs_1_simplex)
                seqs_t_prob = F.softmax(seqs_t_simplex,dim=-1) # (B,L,K)
                seqs_t = sample_from(seqs_t_prob) # (B,L)
                seqs_t = torch.where(batch['generate_mask'],seqs_t,seqs_1)
            else:
                seqs_t = seqs_1.detach().clone()
                seqs_t_simplex = seqs_1_simplex.detach().clone()
                seqs_t_prob = seqs_1_prob.detach().clone()

        # denoise
        pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob  = self.ga_encoder(t, rotmats_t, trans_t_c, angles_t, seqs_t, node_embed, edge_embed, gen_mask, res_mask)
        pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob,dim=-1))
        pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,torch.clamp(seqs_1,0,19))
        pred_trans_1_c,_ = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
        pred_trans_1_c = pred_trans_1 # implicitly enforce zero center in gen_mask, in this way, we dont need to move receptor when sampling

        norm_scale = 1 / (1 - torch.min(t[...,None], torch.tensor(self._interpolant_cfg.t_normalization_clip))) # yim etal.trick, 1/1-t

        # trans vf loss
        trans_loss = torch.sum((pred_trans_1_c - trans_1_c)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        trans_loss = torch.mean(trans_loss)

        # rots vf loss
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        pred_rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        rot_loss = torch.sum(((gt_rot_vf - pred_rot_vf) * norm_scale)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        rot_loss = torch.mean(rot_loss)

        # bb aux loss
        gt_bb_atoms = all_atom.to_atom37(trans_1_c, rotmats_1)[:, :, :3] 
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1_c, pred_rotmats_1)[:, :, :3]
        # gt_bb_atoms = all_atom.to_bb_atoms(trans_1_c, rotmats_1, angles_1[:,:,0]) # N,CA,C,O,CB
        # pred_bb_atoms = all_atom.to_bb_atoms(pred_trans_1_c, pred_rotmats_1, pred_angles_1[:,:,0])
        # print(gt_bb_atoms.shape)
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        bb_atom_loss = torch.mean(bb_atom_loss)
        # bb_atom_loss = torch.mean(torch.where(t[:,0]>=0.75,bb_atom_loss,torch.zeros_like(bb_atom_loss))) # penalty for near gt point

        # seqs vf loss
        seqs_loss = F.cross_entropy(pred_seqs_1_prob.view(-1,pred_seqs_1_prob.shape[-1]),torch.clamp(seqs_1,0,19).view(-1), reduction='none').view(pred_seqs_1_prob.shape[:-1]) # (N,L), not softmax
        seqs_loss = torch.sum(seqs_loss * gen_mask, dim=-1) / (torch.sum(gen_mask,dim=-1) + 1e-8)
        seqs_loss = torch.mean(seqs_loss)

        # we should not use angle mask, as you dont know aa type when generating
        # angle_mask_loss = torch.cat([angle_mask,angle_mask],dim=-1) # (B,L,10)
        # angle vf loss
        angle_mask_loss = torsions_mask.to(batch['aa'].device)
        angle_mask_loss = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
        angle_mask_loss = torch.cat([angle_mask_loss,angle_mask_loss],dim=-1) # (B,L,10)
        angle_mask_loss = torch.logical_and(batch['generate_mask'][...,None].bool(),angle_mask_loss)
        gt_angle_vf = torus.tor_logmap(angles_t, angles_1)
        gt_angle_vf_vec = torch.cat([torch.sin(gt_angle_vf),torch.cos(gt_angle_vf)],dim=-1)
        pred_angle_vf = torus.tor_logmap(angles_t, pred_angles_1)
        pred_angle_vf_vec = torch.cat([torch.sin(pred_angle_vf),torch.cos(pred_angle_vf)],dim=-1)
        # angle_loss = torch.sum(((gt_angle_vf_vec - pred_angle_vf_vec) * norm_scale)**2*gen_mask[...,None],dim=(-1,-2)) / ((torch.sum(gen_mask,dim=-1)) + 1e-8) # (B,)
        angle_loss = torch.sum(((gt_angle_vf_vec - pred_angle_vf_vec) * norm_scale)**2*angle_mask_loss,dim=(-1,-2)) / (torch.sum(angle_mask_loss,dim=(-1,-2)) + 1e-8) # (B,)
        angle_loss = torch.mean(angle_loss)


        # angle aux loss
        angles_1_vec = torch.cat([torch.sin(angles_1),torch.cos(angles_1)],dim=-1)
        pred_angles_1_vec = torch.cat([torch.sin(pred_angles_1),torch.cos(pred_angles_1)],dim=-1)
        # torsion_loss = torch.sum((pred_angles_1_vec - angles_1_vec)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        torsion_loss = torch.sum((pred_angles_1_vec - angles_1_vec)**2*angle_mask_loss,dim=(-1,-2)) / (torch.sum(angle_mask_loss,dim=(-1,-2)) + 1e-8) # (B,)
        torsion_loss = torch.mean(torsion_loss)

        return {
            "trans_loss": trans_loss,
            'rot_loss': rot_loss,
            'bb_atom_loss': bb_atom_loss,
            'seqs_loss': seqs_loss,
            'angle_loss': angle_loss,
            'torsion_loss': torsion_loss,
        }
    
    @torch.no_grad()
    def sample(self, batch, num_steps = 100, sample_bb=True, sample_ang=True, sample_seq=True):

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask = batch['generate_mask'],batch['res_mask']
        K = self._interpolant_cfg.seqs.num_classes
        k = self._interpolant_cfg.seqs.simplex_value
        angle_mask_loss = torsions_mask.to(batch['aa'].device)

        #encode
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch)
        # trans_1_c,center = self.zero_center_part(trans_1,gen_mask,res_mask)
        trans_1_c = trans_1
        seqs_1_simplex = self.seq_to_simplex(seqs_1)
        seqs_1_prob = F.softmax(seqs_1_simplex,dim=-1)

        # # # only sample bb, angle and seq with noise
        # angles_1 = torch.where(batch['generate_mask'][...,None],angles_1,torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype))
        # seqs_1 = torch.where(batch['generate_mask'],seqs_1,torch.randint_like(seqs_1,0,20))
        # seqs_1_simplex = self.seq_to_simplex(seqs_1)
        # seqs_1_prob = F.softmax(seqs_1_simplex,dim=-1)

        #initial noise
        if sample_bb:
            rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
            rotmats_0 = torch.where(batch['generate_mask'][...,None,None],rotmats_0,rotmats_1)
            trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device) # scale with sigma?
            # move center and receptor
            trans_0_c,center = self.zero_center_part(trans_0,gen_mask,res_mask)
            trans_0_c = torch.where(batch['generate_mask'][...,None],trans_0_c,trans_1_c)
        else:
            rotmats_0 = rotmats_1.detach().clone()
            trans_0_c = trans_1_c.detach().clone()
        if sample_ang:
            # angle noise
            angles_0 = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype) # (B,L,5)
            angles_0 = torch.where(batch['generate_mask'][...,None],angles_0,angles_1)
        else:
            angles_0 = angles_1.detach().clone()
        if sample_seq:
            seqs_0_simplex = k * torch.randn((num_batch,num_res,K), device=batch['aa'].device)
            seqs_0_prob = F.softmax(seqs_0_simplex,dim=-1)
            seqs_0 = sample_from(seqs_0_prob)
            seqs_0 = torch.where(batch['generate_mask'],seqs_0,seqs_1)
            seqs_0_simplex = torch.where(batch['generate_mask'][...,None],seqs_0_simplex,seqs_1_simplex)
        else:
            seqs_0 = seqs_1.detach().clone()
            seqs_0_prob = seqs_1_prob.detach().clone()
            seqs_0_simplex = seqs_1_simplex.detach().clone()

        # Set-up time
        ts = torch.linspace(1.e-2, 1.0, num_steps)
        t_1 = ts[0]
        # prot_traj = [{'rotmats':rotmats_0,'trans':trans_0_c,'seqs':seqs_0,'seqs_simplex':seqs_0_simplex,'rotmats_1':rotmats_1,'trans_1':trans_1-center,'seqs_1':seqs_1}]
        clean_traj = []
        rotmats_t_1, trans_t_1_c, angles_t_1, seqs_t_1, seqs_t_1_simplex = rotmats_0, trans_0_c, angles_0, seqs_0, seqs_0_simplex

        # denoise loop
        for t_2 in ts[1:]:
            t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1
            # rots
            pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = self.ga_encoder(t, rotmats_t_1, trans_t_1_c, angles_t_1, seqs_t_1, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
            pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
            # trans, move center
            # pred_trans_1_c,center = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
            pred_trans_1_c = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_1_c) # move receptor also
            # angles
            pred_angles_1 = torch.where(batch['generate_mask'][...,None],pred_angles_1,angles_1)
            # seqs
            pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob,dim=-1))
            pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
            pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
            # seq-angle
            torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
            if not sample_bb:
                pred_trans_1_c = trans_1_c.detach().clone()
                # _,center = self.zero_center_part(trans_1,gen_mask,res_mask)
                pred_rotmats_1 = rotmats_1.detach().clone()
            if not sample_ang:
                pred_angles_1 = angles_1.detach().clone()
            if not sample_seq:
                pred_seqs_1 = seqs_1.detach().clone()
                pred_seqs_1_simplex = seqs_1_simplex.detach().clone()
            clean_traj.append({'rotmats':pred_rotmats_1.cpu(),'trans':pred_trans_1_c.cpu(),'angles':pred_angles_1.cpu(),'seqs':pred_seqs_1.cpu(),'seqs_simplex':pred_seqs_1_simplex.cpu(),
                                    'rotmats_1':rotmats_1.cpu(),'trans_1':trans_1_c.cpu(),'angles_1':angles_1.cpu(),'seqs_1':seqs_1.cpu()})
            # reverse step, also only for gen mask region
            d_t = (t_2-t_1) * torch.ones((num_batch, 1), device=batch['aa'].device)
            # Euler step
            trans_t_2 = trans_t_1_c + (pred_trans_1_c-trans_0_c)*d_t[...,None]
            # trans_t_2_c,center = self.zero_center_part(trans_t_2,gen_mask,res_mask)
            trans_t_2_c = torch.where(batch['generate_mask'][...,None],trans_t_2,trans_1_c) # move receptor also
            # rotmats_t_2 = so3_utils.geodesic_t(d_t[...,None] / (1-t[...,None]), pred_rotmats_1, rotmats_t_1)
            rotmats_t_2 = so3_utils.geodesic_t(d_t[...,None] * 10, pred_rotmats_1, rotmats_t_1)
            rotmats_t_2 = torch.where(batch['generate_mask'][...,None,None],rotmats_t_2,rotmats_1)
            # angles
            angles_t_2 = torus.tor_geodesic_t(d_t[...,None],pred_angles_1, angles_t_1)
            angles_t_2 = torch.where(batch['generate_mask'][...,None],angles_t_2,angles_1)
            # seqs
            seqs_t_2_simplex = seqs_t_1_simplex + (pred_seqs_1_simplex - seqs_0_simplex) * d_t[...,None]
            seqs_t_2 = sample_from(F.softmax(seqs_t_2_simplex,dim=-1))
            seqs_t_2 = torch.where(batch['generate_mask'],seqs_t_2,seqs_1)
            # seq-angle
            torsion_mask = angle_mask_loss[seqs_t_2.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            angles_t_2 = torch.where(torsion_mask.bool(),angles_t_2,torch.zeros_like(angles_t_2))
            
            if not sample_bb:
                trans_t_2_c = trans_1_c.detach().clone()
                rotmats_t_2 = rotmats_1.detach().clone()
            if not sample_ang:
                angles_t_2 = angles_1.detach().clone()
            if not sample_seq:
                seqs_t_2 = seqs_1.detach().clone()
            rotmats_t_1, trans_t_1_c, angles_t_1, seqs_t_1, seqs_t_1_simplex = rotmats_t_2, trans_t_2_c, angles_t_2, seqs_t_2, seqs_t_2_simplex
            t_1 = t_2

        # final step
        t_1 = ts[-1]
        t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1
        pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = self.ga_encoder(t, rotmats_t_1, trans_t_1_c, angles_t_1, seqs_t_1, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
        pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
        # move center
        # pred_trans_1_c,center = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
        pred_trans_1_c = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_1_c) # move receptor also
        # angles
        pred_angles_1 = torch.where(batch['generate_mask'][...,None],pred_angles_1,angles_1)
        # seqs
        pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob,dim=-1))
        pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
        pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
        # seq-angle
        torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
        pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
        if not sample_bb:
            pred_trans_1_c = trans_1_c.detach().clone()
            # _,center = self.zero_center_part(trans_1,gen_mask,res_mask)
            pred_rotmats_1 = rotmats_1.detach().clone()
        if not sample_ang:
            pred_angles_1 = angles_1.detach().clone()
        if not sample_seq:
            pred_seqs_1 = seqs_1.detach().clone()
            pred_seqs_1_simplex = seqs_1_simplex.detach().clone()
        clean_traj.append({'rotmats':pred_rotmats_1.cpu(),'trans':pred_trans_1_c.cpu(),'angles':pred_angles_1.cpu(),'seqs':pred_seqs_1.cpu(),'seqs_simplex':pred_seqs_1_simplex.cpu(),
                                'rotmats_1':rotmats_1.cpu(),'trans_1':trans_1_c.cpu(),'angles_1':angles_1.cpu(),'seqs_1':seqs_1.cpu()})
        
        return clean_traj


# if __name__ == '__main__':
#     prefix_dir = './pepflowww'
#     # config,cfg_name = load_config("../configs/angle/learn_sc.yaml")
#     config,cfg_name = load_config(os.path.join(prefix_dir,"configs/angle/learn_sc.yaml"))
#     # print(config)
#     device = 'cuda:0'
#     dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
#                                             name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate(eight=False), num_workers=4, pin_memory=True)
#     ckpt = torch.load("./checkpoints/600000.pt", map_location=device)
#     seed_all(114514)
#     model = FlowModel(config.model).to(device)
#     model.load_state_dict(process_dic(ckpt['model']))
#     model.eval()
    
#     # print(model)

#     # print(dataset[0]['chain_id'])
#     # print(dataset[0]['id']) 
#     # print(dataset[0]['resseq'])
#     # print(dataset[0]['res_nb'])
#     # print(dataset[0]['icode'])

#     dic = {'id':[],'len':[],'tran':[],'aar':[],'rot':[],'trans_loss':[],'rot_loss':[]}

#     # for batch in tqdm(dataloader):
#     #     batch = recursive_to(batch,device)
#     for i in tqdm(range(len(dataset))):
#         item = dataset[i]
#         data_list = [deepcopy(item) for _ in range(16)]
#         batch = recursive_to(collate_fn(data_list),device)
#         loss_dic = model(batch)
#         # traj_1 = model.sample(batch,num_steps=50,sample_bb=False,sample_ang=True,sample_seq=False)
#         traj_1 = model.sample(batch,num_steps=50,sample_bb=True,sample_ang=True,sample_seq=True)
#         ca_dist = torch.sqrt(torch.sum((traj_1[-1]['trans']-traj_1[-1]['trans_1'])**2*batch['generate_mask'][...,None].cpu().long()) / (torch.sum(batch['generate_mask']) + 1e-8).cpu()) # rmsd
#         rot_dist = torch.sqrt(torch.sum((traj_1[-1]['rotmats']-traj_1[-1]['rotmats_1'])**2*batch['generate_mask'][...,None,None].long().cpu()) / (torch.sum(batch['generate_mask']) + 1e-8).cpu()) # rmsd
#         aar = torch.sum((traj_1[-1]['seqs']==traj_1[-1]['seqs_1']) * batch['generate_mask'].long().cpu()) / (torch.sum(batch['generate_mask']).cpu() + 1e-8)
        

#         print(loss_dic)
#         print(f'tran:{ca_dist},rot:{rot_dist},aar:{aar},len:{batch["generate_mask"].sum().item()}')

#         # free
#         torch.cuda.empty_cache()
#         gc.collect()
        
#     #     dic['tran'].append(ca_dist.item())
#     #     dic['rot'].append(rot_dist.item())
#         dic['aar'].append(aar.item())
#         dic['trans_loss'].append(loss_dic['trans_loss'].item())
#         dic['rot_loss'].append(loss_dic['rot_loss'].item())
#         dic['id'].append(batch['id'][0])
#         dic['len'].append(batch['generate_mask'].sum().item())
#     #     # break

#     #     traj_1[-1]['batch'] = batch
#     #     torch.save(traj_1[-1],f'/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/Data/Models_new/Pack_new/outputs/{batch["id"][0]}.pt')

#         # print(dic)
#     # dic = pd.DataFrame(dic)
#     # dic.to_csv(f'/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/Data/Models_new/Pack/outputs.csv',index=None)

#     print(np.mean(dic['aar']))
#     print(np.mean(dic['trans_loss']))

# if __name__ == '__main__':
#     config,cfg_name = load_config("./configs/angle/learn_angle.yaml")
#     seed_all(114514)
#     device = 'cpu'
#     dataset = PepDataset(structure_dir = config.dataset.train.structure_dir, dataset_dir = config.dataset.train.dataset_dir,
#                                             name = config.dataset.train.name, transform=None, reset=config.dataset.train.reset)
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=PaddingCollate(), num_workers=4, pin_memory=True)
#     model = FlowModel(config.model).to(device)
#     optimizer = torch.optim.Adam(model.parameters(),lr=1.e-4)

#     # ckpt = torch.load('./checkpoints/90000.pt',map_location=device)
#     # model.load_state_dict(process_dic(ckpt['model']))
#     # optimizer.load_state_dict(ckpt['optimizer'])
    
    
#     # torch.autograd.set_detect_anomaly(True)
#     for i,batch in tqdm(enumerate(dataloader)):
#         batch = recursive_to(batch,device)
#         loss_dict = model(batch)
#         loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
#         # if torch.isnan(loss):
#         #     print(i)
#         #     print(batch['id'])

#         loss.backward()
#         orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

#         print(f'{loss_dict},{loss},{orig_grad_norm}')

#         optimizer.step()
#         optimizer.zero_grad()