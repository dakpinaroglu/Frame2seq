import os
from tqdm import tqdm
import torch

from frame2seq.utils import residue_constants
from frame2seq.utils.util import get_neg_pll, read_fasta_file
from frame2seq.utils.pdb2input import get_inference_inputs
from frame2seq.utils.pred2output import output_csv, output_indiv_csv


def score(self, pdb_file, chain_id, fasta_file, save_indiv_neg_pll):
    temperature = 1.0
    seq_mask, aatype, X = get_inference_inputs(pdb_file, chain_id)
    seq_mask = seq_mask.to(self.device)
    aatype = aatype.to(self.device)
    X = X.to(self.device)
    str_form = [residue_constants.ID_TO_AA[int(i)] for i in aatype[0]]
    input_aatype_onehot = residue_constants.sequence_to_onehot(
        sequence=str_form,
        mapping=residue_constants.AA_TO_ID,
    )
    input_aatype_onehot = torch.from_numpy(input_aatype_onehot).float()
    input_aatype_onehot = input_aatype_onehot.unsqueeze(0)
    input_aatype_onehot = input_aatype_onehot.to(self.device)
    input_aatype_onehot = torch.zeros_like(input_aatype_onehot)
    input_aatype_onehot[:, :,
                        20] = 1  # all positions are masked (set to unknown)
    scores, preds = {}, []
    with torch.no_grad():
        pred_seq1 = self.models[0].forward(X, seq_mask, input_aatype_onehot)
        pred_seq2 = self.models[1].forward(X, seq_mask, input_aatype_onehot)
        pred_seq3 = self.models[2].forward(X, seq_mask, input_aatype_onehot)
        pred_seq = (pred_seq1 + pred_seq2 + pred_seq3) / 3  # ensemble
        pred_seq = pred_seq / temperature
        pred_seq = torch.nn.functional.softmax(pred_seq, dim=-1)
        pred_seq = pred_seq[seq_mask]
        if fasta_file is not None:
            input_seqs = read_fasta_file(fasta_file)
            input_seqs = [
                torch.tensor([residue_constants.AA_TO_ID[aa]
                              for aa in seq]).long() for seq in input_seqs
            ]
        else:
            input_seqs = [aatype[seq_mask]]
        for sample in tqdm(range(len(input_seqs))):
            input_seq_i = input_seqs[sample]
            neg_pll, avg_neg_pll = get_neg_pll(pred_seq, input_seq_i)
            scores['pdbid'] = pdb_file.split('/')[-1].split('.')[0]
            scores['chain'] = chain_id
            scores['sample'] = sample
            scores['res_idx'] = [i for i in range(len(input_seq_i))]
            scores['neg_pll'] = [
                neg_pll[i].item() for i in range(len(input_seq_i))
            ]
            scores['avg_neg_pll'] = avg_neg_pll
            input_seq_i = [
                residue_constants.ID_TO_AA[int(i)] for i in input_seq_i
            ]
            input_seq_i = "".join(input_seq_i)
            preds.append([
                scores['pdbid'], scores['chain'], scores['sample'],
                input_seq_i, scores['avg_neg_pll'], temperature
            ])
            csv_dir = os.path.join(self.save_dir, 'scores')
            os.makedirs(csv_dir, exist_ok=True)
            if save_indiv_neg_pll:  # save per-residue negative pseudo-log-likelihoods
                output_indiv_csv(scores, csv_dir)
        output_csv(preds, csv_dir)
