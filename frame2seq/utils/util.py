import torch


def get_neg_pll(probs, seq):
    seq_probs = torch.gather(probs, 1, seq.unsqueeze(-1)).squeeze(-1)
    neg_pll = -1 * torch.log(seq_probs)
    avg_neg_pll = neg_pll.sum().item() / len(neg_pll)
    return neg_pll, avg_neg_pll


def his_tag_mask(aatype_int):
    '''Returns a mask for the histidine tag in the input sequence'''
    his_mask = torch.ones_like(aatype_int)
    if torch.all(aatype_int[0][:3] == 6):
        first_non_his = torch.where(aatype_int[0] != 6)[0][0]
        his_mask[0][:first_non_his] = 0
    if torch.all(aatype_int[0][1:4] == 6):
        first_non_his = torch.where(aatype_int[0][1:] != 6)[0][0] + 1
        his_mask[0][:first_non_his] = 0
    if torch.all(aatype_int[0][-3:] == 6):
        last_non_his = torch.where(aatype_int[0] != 6)[0][-1]
        his_mask[0][last_non_his + 1:] = 0
    return his_mask


def read_fasta_file(fasta_file):
    """
    Read a fasta file and return a list of sequences.
    """
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    sequences = []
    for line in lines:
        if line[0] == '>':
            sequences.append(lines[lines.index(line) + 1].strip())
    return sequences
