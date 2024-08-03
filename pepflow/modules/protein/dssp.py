
from collections import OrderedDict, namedtuple


SSTRUCT_SYMB_TO_INDEX = {'H':0, 'B':1, 'E':2, 'G':3, 'I':4, 'T':5, 'S':6, '-':7}
NONLOOP_SSTRUCT_INDEX = {0,1,2,3,4}
# Taken from: https://jbloomlab.github.io/dms_tools2/dms_tools2.dssp.html
MAX_ASA_TIEN = {
    'A': 129.0, 'C': 167.0, 'D': 193.0, 'E': 223.0, 'F': 240.0, 'G': 104.0, 
    'H': 224.0, 'I': 197.0, 'K': 236.0, 'L': 201.0, 'M': 224.0, 'N': 195.0, 
    'P': 159.0, 'Q': 225.0, 'R': 274.0, 'S': 155.0, 'T': 172.0, 'V': 174.0, 
    'W': 285.0, 'Y': 263.0,
}

DSSPResidueInfo = namedtuple('DSSPResidueInfo', [
    'aa', 'ss', 'acc', 'phi', 'psi', 'index', 'rsa',
])


def secondary_struct_symbol_to_index(s):
    if s in SSTRUCT_SYMB_TO_INDEX:
        return SSTRUCT_SYMB_TO_INDEX[s]
    else:
        return 7


def parse_dssp_file(path):
    with open(path, 'r') as f:
        dssp_dict = make_dssp_dict(f)
    return dssp_dict


def make_dssp_dict(handle):
    """Return a DSSP dictionary, used by mask_dssp_dict (PRIVATE).
    DSSP dictionary maps (chainid, resid) to an amino acid,
    secondary structure symbol, solvent accessibility value, and hydrogen bond
    information (relative dssp indices and hydrogen bond energies) from an open
    DSSP file object.
    Parameters
    ----------
    handle : file or list
        the open DSSP output file handle
        or the list of lines of the DSSP output file
    """
    dssp = OrderedDict()
    start = 0
    for l in handle:
        sl = l.split()
        if len(sl) < 2:
            continue
        if sl[1] == "RESIDUE":
            # Start parsing from here
            start = 1
            continue
        if not start:
            continue
        if l[9] == " ":
            # Skip -- missing residue
            continue

        dssp_index = int(l[:5])
        resseq = int(l[5:10])
        icode = l[10]
        chainid = l[11]
        aa = l[13]
        ss = l[16]
        if ss == " ":
            ss = "-"
        try:
            # NH_O_1_relidx = int(l[38:45])
            # NH_O_1_energy = float(l[46:50])
            # O_NH_1_relidx = int(l[50:56])
            # O_NH_1_energy = float(l[57:61])
            # NH_O_2_relidx = int(l[61:67])
            # NH_O_2_energy = float(l[68:72])
            # O_NH_2_relidx = int(l[72:78])
            # O_NH_2_energy = float(l[79:83])

            acc = int(l[34:38])
            phi = float(l[103:109])
            psi = float(l[109:115])
        except ValueError as exc:
            # DSSP output breaks its own format when there are >9999
            # residues, since only 4 digits are allocated to the seq num
            # field.  See 3kic chain T res 321, 1vsy chain T res 6077.
            # Here, look for whitespace to figure out the number of extra
            # digits, and shift parsing the rest of the line by that amount.
            if l[34] != " ":
                shift = l[34:].find(" ")

                # NH_O_1_relidx = int(l[38 + shift : 45 + shift])
                # NH_O_1_energy = float(l[46 + shift : 50 + shift])
                # O_NH_1_relidx = int(l[50 + shift : 56 + shift])
                # O_NH_1_energy = float(l[57 + shift : 61 + shift])
                # NH_O_2_relidx = int(l[61 + shift : 67 + shift])
                # NH_O_2_energy = float(l[68 + shift : 72 + shift])
                # O_NH_2_relidx = int(l[72 + shift : 78 + shift])
                # O_NH_2_energy = float(l[79 + shift : 83 + shift])

                acc = int(l[34 + shift : 38 + shift])
                phi = float(l[103 + shift : 109 + shift])
                psi = float(l[109 + shift : 115 + shift])
            else:
                raise ValueError(exc) from None
        res_id = (" ", resseq, icode)
        if chainid not in dssp:
            dssp[chainid] = OrderedDict()
        
        if aa in MAX_ASA_TIEN:
            rsa = acc / MAX_ASA_TIEN[aa]
        else:
            rsa = 0.0

        dssp[chainid][res_id] = DSSPResidueInfo(
            index = dssp_index, aa = aa, 
            ss = ss, phi = phi, psi = psi,
            acc = acc, rsa = rsa,
        )
    return dssp


def find_sstruct_ranges(chain_dict, min_length=5):
    sstruct_ranges = []
    start, end = None, None   # start, end
    for i, (res_key, item) in enumerate(chain_dict.items()):
        ss = item.ss
        if secondary_struct_symbol_to_index(ss) in NONLOOP_SSTRUCT_INDEX:
            if start is None: 
                start = end = i
            else: end = i
        else:
            if (start is not None) and (end is not None):
                if (end-start+1) >= min_length:
                    sstruct_ranges.append( (start, end, ) )
                start, end = None, None
    return sstruct_ranges


def find_loop_fragments(chain_dict, min_length=3, max_length=float('inf')):
    ss_ranges = find_sstruct_ranges(chain_dict)
    # print(ss_ranges)
    fragments_all = []
    index_to_reskey = list(chain_dict.keys())
    # for s, e in ss_ranges:
    #     print(index_to_reskey[s], index_to_reskey[e])

    for rng_l, rng_r in zip(ss_ranges[:-1], ss_ranges[1:]):
        start_l, end_l = rng_l
        start_r, end_r = rng_r
        loop_length = start_r - end_l - 1
        if min_length <= loop_length <= max_length:
            loop_reskeys = [index_to_reskey[i] for i in range(end_l+1, start_r)]
            fragments_all.append(loop_reskeys)
    return fragments_all
