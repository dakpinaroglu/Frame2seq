import torch

import frame2seq.utils.protein as protein
from frame2seq.utils.place_ideal_atoms import place_missing_cb, place_missing_o


def read_pdb_string(pdb_file):
    with open(pdb_file) as f:
        pdb_string = f.read()
    return pdb_string


def get_parsed_inputs(pdb_file, chain_id):

    pdb_string = read_pdb_string(pdb_file)
    prot_from_pdb = protein.from_pdb_string(pdb_string, chain_id)
    aatype_int = prot_from_pdb.aatype
    atom_positions = prot_from_pdb.atom_positions
    atom_positions = torch.from_numpy(atom_positions)
    atom_mask = prot_from_pdb.atom_mask
    atom_mask = torch.from_numpy(atom_mask)

    seq_mask = torch.zeros_like(torch.from_numpy(aatype_int))
    seq_mask[aatype_int != 20] = 1

    seq_mask = seq_mask.bool()

    atom_positions = atom_positions[seq_mask]
    aatype_int = aatype_int[seq_mask]
    seq_mask = seq_mask[:len(aatype_int)]

    atom_mask = atom_mask[:, :3].bool()
    atom_mask = atom_mask[:len(aatype_int)]

    missing_bb_idx = [
        i for i in range(len(atom_mask))
        if atom_mask[i][0] == 0 or atom_mask[i][1] == 0 or atom_mask[i][2] == 0
    ]
    seq_mask[missing_bb_idx] = 0
    seq_mask = seq_mask.bool()

    atom_positions = place_missing_cb(atom_positions)
    atom_positions = place_missing_o(atom_positions)
    atom_positions = atom_positions[:, :5, :]

    atom_positions = torch.from_numpy(atom_positions.numpy().round(4))

    return atom_positions, aatype_int, seq_mask
