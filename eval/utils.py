import os
import glob
import pandas as pd
import subprocess
from difflib import SequenceMatcher

from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO, Chain, Select, is_aa
from Bio.PDB.Polypeptide import PPBuilder

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# def parse_pdb_chains(pdb_file):
#     parser = PDBParser()
#     structure = parser.get_structure("protein", pdb_file)
#     pp_builder = PPBuilder()

#     sequences = {}
#     for model in structure:
#         for chain in model:
#             chain_id = chain.get_id()
#             sequence = "".join([str(pp.get_sequence()) for pp in pp_builder.build_peptides(chain)])
#             print(len(sequence))
#             sequences[chain_id] = sequence

#     return sequences

def get_fasta_from_pdb(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("pdb", pdb_file)
    
    fasta_sequence = {}
    for chain in structure.get_chains():
        seq = ""
        for residue in chain.get_residues():
                seq += seq1(residue.get_resname())
        fasta_sequence[chain.id] = seq
    
    return fasta_sequence

def parse_fasta(file):
    sequences = {}
    with open(file, "r") as fasta_file:
        for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
            sequences[i] = str(record.seq).split("/")
    return sequences

def renumber_pdb(input_pdb, output_pdb):
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb)

    chain_dic = {}

    for model in structure:
        old_chains = []
        new_chains = []
        for chain in model: # this may include HEAATM atoms
            new_chain_id = chain.id + "_renum"
            new_chain = Chain.Chain(new_chain_id)
            for i, residue in enumerate(chain):
                new_residue = residue.copy()
                new_residue_id = (residue.id[0], i + 1, residue.id[2])
                new_residue.id = new_residue_id
                new_chain.add(new_residue)
            old_chains.append(chain)
            new_chains.append(new_chain)
            chain_dic[chain.id] = len(list(chain))

        for chain, new_chain in zip(old_chains, new_chains):
            model.detach_child(chain.id)
            new_chain.id = chain.id
            model.add(new_chain)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)

    return chain_dic

def get_chain_dic(input_pdb):
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb)

    chain_dic = {}

    for model in structure:
        for chain in model:
            chain_dic[chain.id] = len([res for res in chain if is_aa(res) and res.has_id('CA')])

    return chain_dic


def keep_backbone_atoms(input_file, output_file):

    class BackboneSelect(Select):
        def accept_atom(self, atom):
            return atom.get_name() in ["N", "CA", "C", "O"]
    
    parser = PDBParser()
    io = PDBIO()

    structure = parser.get_structure("protein", input_file)

    io.set_structure(structure)
    io.save(output_file, BackboneSelect())