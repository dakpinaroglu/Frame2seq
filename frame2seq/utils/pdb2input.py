import torch

from frame2seq.utils.parse import get_parsed_inputs


def get_inference_inputs(pdb_file, chain_id):
    atom_positions, aatype, seq_mask = get_parsed_inputs(pdb_file, chain_id)
    seq_mask = seq_mask.unsqueeze(0)
    aatype = torch.from_numpy(aatype)
    aatype = aatype.unsqueeze(0)
    X = atom_positions
    X = X.unsqueeze(0)
    return seq_mask, aatype, X
