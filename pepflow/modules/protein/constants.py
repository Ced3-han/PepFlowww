import torch
import enum


## others
NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

PAD_RESIDUE_INDEX = 21

##
# Residue identities

non_standard_residue_substitutions = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR',
    'ALA':'ALA', 'CYS':'CYS', 'ASP':'ASP', 'GLU':'GLU', 'PHE':'PHE', 'GLY':'GLY', 'HIS':'HIS', 'ILE':'ILE', 'LYS':'LYS', 'LEU':'LEU',
    'MET':'MET', 'ASN':'ASN', 'PRO':'PRO', 'GLN':'GLN', 'ARG':'ARG', 'SER':'SER', 'THR':'THR', 'VAL':'VAL', 'TRP':'TRP', 'TYR':'TYR',
    'UNK':'UNK'
}


ressymb_to_resindex = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,
}

resindex_to_ressymb = {}
for k,v in ressymb_to_resindex.items(): resindex_to_ressymb[v] = k

BACKBONE_FRAME = 0
OMEGA_FRAME = 1
PHI_FRAME = 2
PSI_FRAME = 3
CHI1_FRAME, CHI2_FRAME, CHI3_FRAME, CHI4_FRAME = 4, 5, 6, 7


class AA(enum.IntEnum):
    ALA = 0; CYS = 1; ASP = 2; GLU = 3; PHE = 4
    GLY = 5; HIS = 6; ILE = 7; LYS = 8; LEU = 9
    MET = 10; ASN = 11; PRO = 12; GLN = 13; ARG = 14
    SER = 15; THR = 16; VAL = 17; TRP = 18; TYR = 19
    UNK = 20

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and len(value) == 3:      # three representation
            if value in non_standard_residue_substitutions:
                value = non_standard_residue_substitutions[value]
            if value in cls._member_names_:
                return getattr(cls, value)
        elif isinstance(value, str) and len(value) == 1:    # one representation
            if value in ressymb_to_resindex:
                return cls(ressymb_to_resindex[value])

        return super()._missing_(value)

    def __str__(self):
        return self.name

    @classmethod
    def is_aa(cls, value):
        return (value in ressymb_to_resindex) or \
            (value in non_standard_residue_substitutions) or \
            (value in cls._member_names_)


num_aa_types = len(AA)

##
# Atom identities

class BBHeavyAtom(enum.IntEnum):
    N = 0; CA = 1; C = 2; O = 3; CB = 4; OXT=14;

max_num_heavyatoms = 15
max_num_hydrogens = 16
max_num_allatoms = max_num_heavyatoms + max_num_hydrogens

restype_to_heavyatom_names = {
    AA.ALA: ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.ARG: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    '', 'OXT'],
    AA.ASN: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    '', 'OXT'],
    AA.ASP: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    '', 'OXT'],
    AA.CYS: ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.GLN: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    '', 'OXT'],
    AA.GLU: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    '', 'OXT'],
    AA.GLY: ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.HIS: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    '', 'OXT'],
    AA.ILE: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    '', 'OXT'],
    AA.LEU: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    '', 'OXT'],
    AA.LYS: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    '', 'OXT'],
    AA.MET: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    '', 'OXT'],
    AA.PHE: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    '', 'OXT'],
    AA.PRO: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.SER: ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.THR: ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.TRP: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT'],
    AA.TYR: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    '', 'OXT'],
    AA.VAL: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.UNK: ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
}
for names in restype_to_heavyatom_names.values(): assert len(names) == max_num_heavyatoms

restype_to_hydrogen_names = {
    AA.ALA: ['H', 'H2', 'H3', 'HA', 'HB1', 'HB2', 'HB3', 'HXT', '', '', '', '', '', '', '', ''],
    AA.CYS: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG', 'HXT', '', '', '', '', '', '', '', ''],
    AA.ASP: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD2', 'HXT', '', '', '', '', '', '', '', ''],
    AA.GLU: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HE2', 'HXT', '', '', '', '', '', ''],
    AA.PHE: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ', 'HXT', '', '', '', ''],
    AA.GLY: ['H', 'H2', 'H3', 'HA2', 'HA3', 'HXT', '', '', '', '', '', '', '', '', '', ''],
    AA.HIS: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HXT', '', '', '', '', ''],
    AA.ILE: ['H', 'H2', 'H3', 'HA', 'HB', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HD11', 'HD12', 'HD13', 'HXT', '', ''],
    AA.LYS: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3', 'HXT'],
    AA.LEU: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23', 'HXT', '', ''],
    AA.MET: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HE1', 'HE2', 'HE3', 'HXT', '', '', '', ''],
    AA.ASN: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD21', 'HD22', 'HXT', '', '', '', '', '', '', ''],
    AA.PRO: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HXT', '', '', '', '', ''],
    AA.GLN: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HE21', 'HE22', 'HXT', '', '', '', '', ''],
    AA.ARG: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE', 'HH11', 'HH12', 'HH21', 'HH22', 'HXT'],
    AA.SER: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HG', 'HXT', '', '', '', '', '', '', '', ''],
    AA.THR: ['H', 'H2', 'H3', 'HA', 'HB', 'HG1', 'HG21', 'HG22', 'HG23', 'HXT', '', '', '', '', '', ''],
    AA.VAL: ['H', 'H2', 'H3', 'HA', 'HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HXT', '', '', '', ''],
    AA.TRP: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2', 'HXT', '', '', ''],
    AA.TYR: ['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HH', 'HXT', '', '', '', ''],
    AA.UNK: ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
}
for names in restype_to_hydrogen_names.values(): assert len(names) == max_num_hydrogens

restype_to_allatom_names = {
    restype: restype_to_heavyatom_names[restype] + restype_to_hydrogen_names[restype]
    for restype in AA
}

restype_atom14_name_to_index = {
    resname: {name: index for index, name in enumerate(atoms) if name != ""}
    for resname, atoms in restype_to_heavyatom_names.items()
}

##
# Bond identities

class BondType(enum.IntEnum):
    NoBond = 0
    Single = 1
    Double = 2
    Triple = 3
    AromaticSingle = 5
    AromaticDouble = 6

BT = BondType
restype_to_bonded_atom_name_pairs = {
    AA.ALA: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'HB1', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.CYS: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'SG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('SG', 'HG', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.ASP: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'OD1', BT.AromaticDouble), ('CG', 'OD2', BT.AromaticSingle), ('OD2', 'HD2', BT.AromaticSingle), 
        ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.GLU: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle), 
        ('CD', 'OE1', BT.AromaticDouble), ('CD', 'OE2', BT.AromaticSingle), ('OE2', 'HE2', BT.AromaticSingle), 
        ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.PHE: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'CD1', BT.AromaticDouble), ('CG', 'CD2', BT.AromaticSingle), ('CD1', 'CE1', BT.AromaticSingle), 
        ('CD1', 'HD1', BT.AromaticSingle), ('CD2', 'CE2', BT.AromaticDouble), ('CD2', 'HD2', BT.AromaticSingle), 
        ('CE1', 'CZ', BT.AromaticDouble), ('CE1', 'HE1', BT.AromaticSingle), ('CE2', 'CZ', BT.AromaticSingle), 
        ('CE2', 'HE2', BT.AromaticSingle), ('CZ', 'HZ', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.GLY: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'HA2', BT.AromaticSingle), 
        ('CA', 'HA3', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.HIS: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'ND1', BT.AromaticSingle), ('CG', 'CD2', BT.AromaticDouble), ('ND1', 'CE1', BT.AromaticDouble), 
        ('ND1', 'HD1', BT.AromaticSingle), ('CD2', 'NE2', BT.AromaticSingle), ('CD2', 'HD2', BT.AromaticSingle), 
        ('CE1', 'NE2', BT.AromaticSingle), ('CE1', 'HE1', BT.AromaticSingle), ('NE2', 'HE2', BT.AromaticSingle), 
        ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.ILE: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG1', BT.AromaticSingle), ('CB', 'CG2', BT.AromaticSingle), ('CB', 'HB', BT.AromaticSingle), 
        ('CG1', 'CD1', BT.AromaticSingle), ('CG1', 'HG12', BT.AromaticSingle), ('CG1', 'HG13', BT.AromaticSingle), 
        ('CG2', 'HG21', BT.AromaticSingle), ('CG2', 'HG22', BT.AromaticSingle), ('CG2', 'HG23', BT.AromaticSingle), 
        ('CD1', 'HD11', BT.AromaticSingle), ('CD1', 'HD12', BT.AromaticSingle), ('CD1', 'HD13', BT.AromaticSingle), 
        ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.LYS: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle), 
        ('CD', 'CE', BT.AromaticSingle), ('CD', 'HD2', BT.AromaticSingle), ('CD', 'HD3', BT.AromaticSingle), 
        ('CE', 'NZ', BT.AromaticSingle), ('CE', 'HE2', BT.AromaticSingle), ('CE', 'HE3', BT.AromaticSingle), 
        ('NZ', 'HZ1', BT.AromaticSingle), ('NZ', 'HZ2', BT.AromaticSingle), ('NZ', 'HZ3', BT.AromaticSingle), 
        ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.LEU: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'CD1', BT.AromaticSingle), ('CG', 'CD2', BT.AromaticSingle), ('CG', 'HG', BT.AromaticSingle), 
        ('CD1', 'HD11', BT.AromaticSingle), ('CD1', 'HD12', BT.AromaticSingle), ('CD1', 'HD13', BT.AromaticSingle), 
        ('CD2', 'HD21', BT.AromaticSingle), ('CD2', 'HD22', BT.AromaticSingle), ('CD2', 'HD23', BT.AromaticSingle), 
        ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.MET: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'SD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle), 
        ('SD', 'CE', BT.AromaticSingle), ('CE', 'HE1', BT.AromaticSingle), ('CE', 'HE2', BT.AromaticSingle), 
        ('CE', 'HE3', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.ASN: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'OD1', BT.AromaticDouble), ('CG', 'ND2', BT.AromaticSingle), ('ND2', 'HD21', BT.AromaticSingle), 
        ('ND2', 'HD22', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.PRO: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('N', 'CD', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), 
        ('CA', 'CB', BT.AromaticSingle), ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), 
        ('C', 'OXT', BT.AromaticSingle), ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), 
        ('CB', 'HB3', BT.AromaticSingle), ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), 
        ('CG', 'HG3', BT.AromaticSingle), ('CD', 'HD2', BT.AromaticSingle), ('CD', 'HD3', BT.AromaticSingle), 
        ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.GLN: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle), 
        ('CD', 'OE1', BT.AromaticDouble), ('CD', 'NE2', BT.AromaticSingle), ('NE2', 'HE21', BT.AromaticSingle), 
        ('NE2', 'HE22', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.ARG: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'CD', BT.AromaticSingle), ('CG', 'HG2', BT.AromaticSingle), ('CG', 'HG3', BT.AromaticSingle), 
        ('CD', 'NE', BT.AromaticSingle), ('CD', 'HD2', BT.AromaticSingle), ('CD', 'HD3', BT.AromaticSingle), 
        ('NE', 'CZ', BT.AromaticSingle), ('NE', 'HE', BT.AromaticSingle), ('CZ', 'NH1', BT.AromaticSingle), 
        ('CZ', 'NH2', BT.AromaticDouble), ('NH1', 'HH11', BT.AromaticSingle), ('NH1', 'HH12', BT.AromaticSingle), 
        ('NH2', 'HH21', BT.AromaticSingle), ('NH2', 'HH22', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.SER: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'OG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('OG', 'HG', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.THR: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'OG1', BT.AromaticSingle), ('CB', 'CG2', BT.AromaticSingle), ('CB', 'HB', BT.AromaticSingle), 
        ('OG1', 'HG1', BT.AromaticSingle), ('CG2', 'HG21', BT.AromaticSingle), ('CG2', 'HG22', BT.AromaticSingle), 
        ('CG2', 'HG23', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.VAL: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG1', BT.AromaticSingle), ('CB', 'CG2', BT.AromaticSingle), ('CB', 'HB', BT.AromaticSingle), 
        ('CG1', 'HG11', BT.AromaticSingle), ('CG1', 'HG12', BT.AromaticSingle), ('CG1', 'HG13', BT.AromaticSingle), 
        ('CG2', 'HG21', BT.AromaticSingle), ('CG2', 'HG22', BT.AromaticSingle), ('CG2', 'HG23', BT.AromaticSingle), 
        ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.TRP: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'CD1', BT.AromaticDouble), ('CG', 'CD2', BT.AromaticSingle), ('CD1', 'NE1', BT.AromaticSingle), 
        ('CD1', 'HD1', BT.AromaticSingle), ('CD2', 'CE2', BT.AromaticDouble), ('CD2', 'CE3', BT.AromaticSingle), 
        ('NE1', 'CE2', BT.AromaticSingle), ('NE1', 'HE1', BT.AromaticSingle), ('CE2', 'CZ2', BT.AromaticSingle), 
        ('CE3', 'CZ3', BT.AromaticDouble), ('CE3', 'HE3', BT.AromaticSingle), ('CZ2', 'CH2', BT.AromaticDouble), 
        ('CZ2', 'HZ2', BT.AromaticSingle), ('CZ3', 'CH2', BT.AromaticSingle), ('CZ3', 'HZ3', BT.AromaticSingle), 
        ('CH2', 'HH2', BT.AromaticSingle), ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.TYR: [
        ('N', 'H', BT.Single), ('N', 'H2', BT.Single), ('N', 'H3', BT.Single), 
        ('N', 'CA', BT.AromaticSingle), ('CA', 'C', BT.AromaticSingle), ('CA', 'CB', BT.AromaticSingle), 
        ('CA', 'HA', BT.AromaticSingle), ('C', 'O', BT.AromaticDouble), ('C', 'OXT', BT.AromaticSingle), 
        ('CB', 'CG', BT.AromaticSingle), ('CB', 'HB2', BT.AromaticSingle), ('CB', 'HB3', BT.AromaticSingle), 
        ('CG', 'CD1', BT.AromaticDouble), ('CG', 'CD2', BT.AromaticSingle), ('CD1', 'CE1', BT.AromaticSingle), 
        ('CD1', 'HD1', BT.AromaticSingle), ('CD2', 'CE2', BT.AromaticDouble), ('CD2', 'HD2', BT.AromaticSingle), 
        ('CE1', 'CZ', BT.AromaticDouble), ('CE1', 'HE1', BT.AromaticSingle), ('CE2', 'CZ', BT.AromaticSingle), 
        ('CE2', 'HE2', BT.AromaticSingle), ('CZ', 'OH', BT.AromaticSingle), ('OH', 'HH', BT.AromaticSingle), 
        ('OXT', 'HXT', BT.AromaticSingle), ], 
    AA.UNK: [], 
}


restype_to_allatom_bond_matrix = {
    restype: torch.zeros([max_num_allatoms, max_num_allatoms], dtype=torch.long)
    for restype in AA
}
restype_to_heavyatom_bond_matrix = {
    restype: torch.zeros([max_num_heavyatoms, max_num_heavyatoms], dtype=torch.long)
    for restype in AA
}

def _make_bond_matrices():
    for restype in AA:
        for atom1_name, atom2_name, bond_type in restype_to_bonded_atom_name_pairs[restype]:
            idx1 = restype_to_allatom_names[restype].index(atom1_name)
            idx2 = restype_to_allatom_names[restype].index(atom2_name)
            restype_to_allatom_bond_matrix[restype][idx1, idx2] = bond_type
            restype_to_allatom_bond_matrix[restype][idx2, idx1] = bond_type
            if atom1_name in restype_to_heavyatom_names[restype] and \
               atom2_name in restype_to_heavyatom_names[restype]:
                jdx1 = restype_to_heavyatom_names[restype].index(atom1_name)
                jdx2 = restype_to_heavyatom_names[restype].index(atom2_name)
                restype_to_heavyatom_bond_matrix[restype][jdx1, jdx2] = bond_type
                restype_to_heavyatom_bond_matrix[restype][jdx2, jdx1] = bond_type
_make_bond_matrices()


##
# Torsion geometry and ideal coordinates

class Torsion(enum.IntEnum):
    Backbone = 0
    Omega = 1
    Phi = 2
    Psi = 3
    Chi1 = 4
    Chi2 = 5
    Chi3 = 6
    Chi7 = 7


chi_angles_atoms = {
    AA.ALA: [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    AA.ARG: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    AA.ASN: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    AA.ASP: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    AA.CYS: [['N', 'CA', 'CB', 'SG']],
    AA.GLN: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    AA.GLU: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    AA.GLY: [],
    AA.HIS: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    AA.ILE: [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    AA.LEU: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.LYS: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    AA.MET: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    AA.PHE: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.PRO: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    AA.SER: [['N', 'CA', 'CB', 'OG']],
    AA.THR: [['N', 'CA', 'CB', 'OG1']],
    AA.TRP: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.TYR: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.VAL: [['N', 'CA', 'CB', 'CG1']],
}


chi_angles_mask = {
    AA.ALA: [False, False, False, False],  # ALA
    AA.ARG: [True , True , True , True ],  # ARG
    AA.ASN: [True , True , False, False],  # ASN
    AA.ASP: [True , True , False, False],  # ASP
    AA.CYS: [True , False, False, False],  # CYS
    AA.GLN: [True , True , True , False],  # GLN
    AA.GLU: [True , True , True , False],  # GLU
    AA.GLY: [False, False, False, False],  # GLY
    AA.HIS: [True , True , False, False],  # HIS
    AA.ILE: [True , True , False, False],  # ILE
    AA.LEU: [True , True , False, False],  # LEU
    AA.LYS: [True , True , True , True ],  # LYS
    AA.MET: [True , True , True , False],  # MET
    AA.PHE: [True , True , False, False],  # PHE
    AA.PRO: [True , True , False, False],  # PRO
    AA.SER: [True , False, False, False],  # SER
    AA.THR: [True , False, False, False],  # THR
    AA.TRP: [True , True , False, False],  # TRP
    AA.TYR: [True , True , False, False],  # TYR
    AA.VAL: [True , False, False, False],  # VAL
    AA.UNK: [False, False, False, False],  # UNK
}


chi_pi_periodic = {
    AA.ALA: [False, False, False, False],  # ALA
    AA.ARG: [False, False, False, False],  # ARG
    AA.ASN: [False, False, False, False],  # ASN
    AA.ASP: [False, True , False, False],  # ASP
    AA.CYS: [False, False, False, False],  # CYS
    AA.GLN: [False, False, False, False],  # GLN
    AA.GLU: [False, False, True , False],  # GLU
    AA.GLY: [False, False, False, False],  # GLY
    AA.HIS: [False, False, False, False],  # HIS
    AA.ILE: [False, False, False, False],  # ILE
    AA.LEU: [False, False, False, False],  # LEU
    AA.LYS: [False, False, False, False],  # LYS
    AA.MET: [False, False, False, False],  # MET
    AA.PHE: [False, True , False, False],  # PHE
    AA.PRO: [False, False, False, False],  # PRO
    AA.SER: [False, False, False, False],  # SER
    AA.THR: [False, False, False, False],  # THR
    AA.TRP: [False, False, False, False],  # TRP
    AA.TYR: [False, True , False, False],  # TYR
    AA.VAL: [False, False, False, False],  # VAL
    AA.UNK: [False, False, False, False],  # UNK
}


rigid_group_heavy_atom_positions = {
    AA.ALA: [
        ['N', 0, (-0.525, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.529, -0.774, -1.205)],
        ['O', 3, (0.627, 1.062, 0.000)],
    ],
    AA.ARG: [
        ['N', 0, (-0.524, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.524, -0.778, -1.209)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.616, 1.390, -0.000)],
        ['CD', 5, (0.564, 1.414, 0.000)],
        ['NE', 6, (0.539, 1.357, -0.000)],
        ['NH1', 7, (0.206, 2.301, 0.000)],
        ['NH2', 7, (2.078, 0.978, -0.000)],
        ['CZ', 7, (0.758, 1.093, -0.000)],
    ],
    AA.ASN: [
        ['N', 0, (-0.536, 1.357, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.531, -0.787, -1.200)],
        ['O', 3, (0.625, 1.062, 0.000)],
        ['CG', 4, (0.584, 1.399, 0.000)],
        ['ND2', 5, (0.593, -1.188, 0.001)],
        ['OD1', 5, (0.633, 1.059, 0.000)],
    ],
    AA.ASP: [
        ['N', 0, (-0.525, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, 0.000, -0.000)],
        ['CB', 0, (-0.526, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.593, 1.398, -0.000)],
        ['OD1', 5, (0.610, 1.091, 0.000)],
        ['OD2', 5, (0.592, -1.101, -0.003)],
    ],
    AA.CYS: [
        ['N', 0, (-0.522, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, 0.000)],
        ['CB', 0, (-0.519, -0.773, -1.212)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['SG', 4, (0.728, 1.653, 0.000)],
    ],
    AA.GLN: [
        ['N', 0, (-0.526, 1.361, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.779, -1.207)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.615, 1.393, 0.000)],
        ['CD', 5, (0.587, 1.399, -0.000)],
        ['NE2', 6, (0.593, -1.189, -0.001)],
        ['OE1', 6, (0.634, 1.060, 0.000)],
    ],
    AA.GLU: [
        ['N', 0, (-0.528, 1.361, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.526, -0.781, -1.207)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.615, 1.392, 0.000)],
        ['CD', 5, (0.600, 1.397, 0.000)],
        ['OE1', 6, (0.607, 1.095, -0.000)],
        ['OE2', 6, (0.589, -1.104, -0.001)],
    ],
    AA.GLY: [
        ['N', 0, (-0.572, 1.337, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.517, -0.000, -0.000)],
        ['O', 3, (0.626, 1.062, -0.000)],
    ],
    AA.HIS: [
        ['N', 0, (-0.527, 1.360, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.778, -1.208)],
        ['O', 3, (0.625, 1.063, 0.000)],
        ['CG', 4, (0.600, 1.370, -0.000)],
        ['CD2', 5, (0.889, -1.021, 0.003)],
        ['ND1', 5, (0.744, 1.160, -0.000)],
        ['CE1', 5, (2.030, 0.851, 0.002)],
        ['NE2', 5, (2.145, -0.466, 0.004)],
    ],
    AA.ILE: [
        ['N', 0, (-0.493, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.536, -0.793, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.534, 1.437, -0.000)],
        ['CG2', 4, (0.540, -0.785, -1.199)],
        ['CD1', 5, (0.619, 1.391, 0.000)],
    ],
    AA.LEU: [
        ['N', 0, (-0.520, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.773, -1.214)],
        ['O', 3, (0.625, 1.063, -0.000)],
        ['CG', 4, (0.678, 1.371, 0.000)],
        ['CD1', 5, (0.530, 1.430, -0.000)],
        ['CD2', 5, (0.535, -0.774, 1.200)],
    ],
    AA.LYS: [
        ['N', 0, (-0.526, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.524, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.619, 1.390, 0.000)],
        ['CD', 5, (0.559, 1.417, 0.000)],
        ['CE', 6, (0.560, 1.416, 0.000)],
        ['NZ', 7, (0.554, 1.387, 0.000)],
    ],
    AA.MET: [
        ['N', 0, (-0.521, 1.364, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.210)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['CG', 4, (0.613, 1.391, -0.000)],
        ['SD', 5, (0.703, 1.695, 0.000)],
        ['CE', 6, (0.320, 1.786, -0.000)],
    ],
    AA.PHE: [
        ['N', 0, (-0.518, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, -0.000)],
        ['CB', 0, (-0.525, -0.776, -1.212)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.377, 0.000)],
        ['CD1', 5, (0.709, 1.195, -0.000)],
        ['CD2', 5, (0.706, -1.196, 0.000)],
        ['CE1', 5, (2.102, 1.198, -0.000)],
        ['CE2', 5, (2.098, -1.201, -0.000)],
        ['CZ', 5, (2.794, -0.003, -0.001)],
    ],
    AA.PRO: [
        ['N', 0, (-0.566, 1.351, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, 0.000)],
        ['CB', 0, (-0.546, -0.611, -1.293)],
        ['O', 3, (0.621, 1.066, 0.000)],
        ['CG', 4, (0.382, 1.445, 0.0)],
        # ['CD', 5, (0.427, 1.440, 0.0)],
        ['CD', 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    AA.SER: [
        ['N', 0, (-0.529, 1.360, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.518, -0.777, -1.211)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['OG', 4, (0.503, 1.325, 0.000)],
    ],
    AA.THR: [
        ['N', 0, (-0.517, 1.364, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, -0.000)],
        ['CB', 0, (-0.516, -0.793, -1.215)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG2', 4, (0.550, -0.718, -1.228)],
        ['OG1', 4, (0.472, 1.353, 0.000)],
    ],
    AA.TRP: [
        ['N', 0, (-0.521, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.212)],
        ['O', 3, (0.627, 1.062, 0.000)],
        ['CG', 4, (0.609, 1.370, -0.000)],
        ['CD1', 5, (0.824, 1.091, 0.000)],
        ['CD2', 5, (0.854, -1.148, -0.005)],
        ['CE2', 5, (2.186, -0.678, -0.007)],
        ['CE3', 5, (0.622, -2.530, -0.007)],
        ['NE1', 5, (2.140, 0.690, -0.004)],
        ['CH2', 5, (3.028, -2.890, -0.013)],
        ['CZ2', 5, (3.283, -1.543, -0.011)],
        ['CZ3', 5, (1.715, -3.389, -0.011)],
    ],
    AA.TYR: [
        ['N', 0, (-0.522, 1.362, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.776, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.382, -0.000)],
        ['CD1', 5, (0.716, 1.195, -0.000)],
        ['CD2', 5, (0.713, -1.194, -0.001)],
        ['CE1', 5, (2.107, 1.200, -0.002)],
        ['CE2', 5, (2.104, -1.201, -0.003)],
        ['OH', 5, (4.168, -0.002, -0.005)],
        ['CZ', 5, (2.791, -0.001, -0.003)],
    ],
    AA.VAL: [
        ['N', 0, (-0.494, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.533, -0.795, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.540, 1.429, -0.000)],
        ['CG2', 4, (0.533, -0.776, 1.203)],
    ],
}


# The following tensors are initialized by `_make_rigid_group_constants`
restype_rigid_group_rotation = torch.zeros([21, 8, 3, 3])
restype_rigid_group_translation = torch.zeros([21, 8, 3])
restype_heavyatom_to_rigid_group = torch.zeros([21, 14], dtype=torch.long)
restype_heavyatom_rigid_group_positions = torch.zeros([21, 14, 3])

def _make_rigid_group_constants():

    def _make_rotation_matrix(ex, ey):
        ex_normalized = ex / torch.linalg.norm(ex)

        # make ey perpendicular to ex
        ey_normalized = ey - torch.dot(ey, ex_normalized) * ex_normalized
        ey_normalized /= torch.linalg.norm(ey_normalized)

        eznorm = torch.cross(ex_normalized, ey_normalized)
        m = torch.stack([ex_normalized, ey_normalized, eznorm]).transpose(0, 1) # (3, 3_index)
        return m

    for restype in AA:
        if restype == AA.UNK: continue

        atom_groups = {
            name: group 
            for name, group, _ in rigid_group_heavy_atom_positions[restype]
        }
        atom_positions = {
            name: torch.FloatTensor(pos) 
            for name, _, pos in rigid_group_heavy_atom_positions[restype]
        }

        # Atom 14 rigid group positions
        for atom_idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
            if (atom_name == '') or (atom_name not in atom_groups): continue
            restype_heavyatom_to_rigid_group[restype, atom_idx] = atom_groups[atom_name]
            restype_heavyatom_rigid_group_positions[restype, atom_idx, :] = atom_positions[atom_name]

        # 0: backbone to backbone
        restype_rigid_group_rotation[restype, Torsion.Backbone, :, :] = torch.eye(3)
        restype_rigid_group_translation[restype, Torsion.Backbone, :] = torch.zeros([3])

        # 1: omega-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Omega, :, :] = torch.eye(3)
        restype_rigid_group_translation[restype, Torsion.Omega, :] = torch.zeros([3])

        # 2: phi-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Phi, :, :] = _make_rotation_matrix(
            ex = atom_positions['N'] - atom_positions['CA'],
            ey = torch.FloatTensor([1., 0., 0.]),
        )
        restype_rigid_group_translation[restype, Torsion.Phi, :] = atom_positions['N']

        # 3: psi-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Psi, :, :] = _make_rotation_matrix(
            ex = atom_positions['C'] - atom_positions['CA'],
            ey = atom_positions['CA'] - atom_positions['N'],    # In accordance to the definition of psi angle
        )
        restype_rigid_group_translation[restype, Torsion.Psi, :] = atom_positions['C']

        # 4: chi1-frame to backbone
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[restype][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            restype_rigid_group_rotation[restype, Torsion.Chi1, :, :] = _make_rotation_matrix(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
            )
            restype_rigid_group_translation[restype, Torsion.Chi1, :] = base_atom_positions[2]

        # chi2-chi1
        # chi3-chi2
        # chi4-chi3
        for chi_idx in range(1, 4):
            if chi_angles_mask[restype][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[restype][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                restype_rigid_group_rotation[restype, Torsion.Chi1+chi_idx, :, :] = _make_rotation_matrix(
                    ex = axis_end_atom_position,
                    ey = torch.FloatTensor([-1., 0., 0.]),
                )
                restype_rigid_group_translation[restype, Torsion.Chi1+chi_idx, :] = axis_end_atom_position

_make_rigid_group_constants()


"""
# The following tensors are taken from diffab
"""
backbone_atom_coordinates = {
    AA.ALA: [
        (-0.525, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA.ARG: [
        (-0.524, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA.ASN: [
        (-0.536, 1.357, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA.ASP: [
        (-0.525, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, 0.0, -0.0),  # C
    ],
    AA.CYS: [
        (-0.522, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, 0.0, 0.0),  # C
    ],
    AA.GLN: [
        (-0.526, 1.361, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, 0.0),  # C
    ],
    AA.GLU: [
        (-0.528, 1.361, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA.GLY: [
        (-0.572, 1.337, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.517, -0.0, -0.0),  # C
    ],
    AA.HIS: [
        (-0.527, 1.36, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, 0.0, 0.0),  # C
    ],
    AA.ILE: [
        (-0.493, 1.373, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, -0.0),  # C
    ],
    AA.LEU: [
        (-0.52, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA.LYS: [
        (-0.526, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, 0.0),  # C
    ],
    AA.MET: [
        (-0.521, 1.364, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, 0.0, 0.0),  # C
    ],
    AA.PHE: [
        (-0.518, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, 0.0, -0.0),  # C
    ],
    AA.PRO: [
        (-0.566, 1.351, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, 0.0),  # C
    ],
    AA.SER: [
        (-0.529, 1.36, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA.THR: [
        (-0.517, 1.364, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, -0.0),  # C
    ],
    AA.TRP: [
        (-0.521, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, 0.0),  # C
    ],
    AA.TYR: [
        (-0.522, 1.362, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, -0.0, -0.0),  # C
    ],
    AA.VAL: [
        (-0.494, 1.373, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, -0.0),  # C
    ],
}

bb_oxygen_coordinate = {
    AA.ALA: (2.153, -1.062, 0.0),
    AA.ARG: (2.151, -1.062, 0.0),
    AA.ASN: (2.151, -1.062, 0.0),
    AA.ASP: (2.153, -1.062, 0.0),
    AA.CYS: (2.149, -1.062, 0.0),
    AA.GLN: (2.152, -1.062, 0.0),
    AA.GLU: (2.152, -1.062, 0.0),
    AA.GLY: (2.143, -1.062, 0.0),
    AA.HIS: (2.15, -1.063, 0.0),
    AA.ILE: (2.154, -1.062, 0.0),
    AA.LEU: (2.15, -1.063, 0.0),
    AA.LYS: (2.152, -1.062, 0.0),
    AA.MET: (2.15, -1.062, 0.0),
    AA.PHE: (2.15, -1.062, 0.0),
    AA.PRO: (2.148, -1.066, 0.0),
    AA.SER: (2.151, -1.062, 0.0),
    AA.THR: (2.152, -1.062, 0.0),
    AA.TRP: (2.152, -1.062, 0.0),
    AA.TYR: (2.151, -1.062, 0.0),
    AA.VAL: (2.154, -1.062, 0.0),
}

backbone_atom_coordinates_tensor = torch.zeros([21, 3, 3])
bb_oxygen_coordinate_tensor = torch.zeros([21, 3])

def make_coordinate_tensors():
    for restype, atom_coords in backbone_atom_coordinates.items():
        for atom_id, atom_coord in enumerate(atom_coords):
            backbone_atom_coordinates_tensor[restype][atom_id] = torch.FloatTensor(atom_coord)
    
    for restype, bb_oxy_coord in bb_oxygen_coordinate.items():
        bb_oxygen_coordinate_tensor[restype] = torch.FloatTensor(bb_oxy_coord)
make_coordinate_tensors()
