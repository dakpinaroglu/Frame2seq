import os
from tqdm import tqdm
import torch

from frame2seq.utils import residue_constants
from frame2seq.utils.util import get_neg_pll
from frame2seq.utils.pdb2input import get_inference_inputs
from frame2seq.utils.pred2output import output_fasta, output_csv


def design(self, pdb_file, chain_id, temperature, num_samples, save_neg_pll,
           verbose):
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
    model_outs, scores = {}, {}
    with torch.no_grad():
        pred_seq1 = self.models[0].forward(X, seq_mask, input_aatype_onehot)
        pred_seq2 = self.models[1].forward(X, seq_mask, input_aatype_onehot)
        pred_seq3 = self.models[2].forward(X, seq_mask, input_aatype_onehot)
        pred_seq = (pred_seq1 + pred_seq2 + pred_seq3) / 3  # ensemble
        pred_seq = pred_seq / temperature
        pred_seq = torch.nn.functional.softmax(pred_seq, dim=-1)
        pred_seq = pred_seq[seq_mask]
        sampled_seq = torch.multinomial(pred_seq,
                                        num_samples=num_samples,
                                        replacement=True)
        for sample in tqdm(range(num_samples)):
            sampled_seq_i = sampled_seq[:, sample]
            neg_pll, avg_neg_pll = get_neg_pll(pred_seq, sampled_seq_i)
            recovery = torch.sum(
                sampled_seq_i == aatype[seq_mask]) / torch.sum(seq_mask)
            sampled_seq_i = [
                residue_constants.ID_TO_AA[int(i)] for i in sampled_seq_i
            ]
            sampled_seq_i = "".join(sampled_seq_i)
            if verbose:
                print(f"Recovery : {recovery*100:.2f}%")
                print(
                    f"Average negative pseudo-log-likelihood : {avg_neg_pll:.2f}"
                )
                print(f"Sequence: {sampled_seq_i}")
            model_outs['pdbid'] = pdb_file.split('/')[-1].split('.')[0]
            model_outs['chain'] = chain_id
            model_outs['sample'] = sample
            model_outs['seq'] = sampled_seq_i
            model_outs['recovery'] = recovery
            model_outs['avg_neg_pll'] = avg_neg_pll
            model_outs['temp'] = temperature
            fasta_dir = os.path.join(self.save_dir, 'seqs')
            os.makedirs(fasta_dir, exist_ok=True)
            output_fasta(model_outs, fasta_dir)
            if save_neg_pll:  # save per-residue negative pseudo-log-likelihoods
                scores['pdbid'] = pdb_file.split('/')[-1].split('.')[0]
                scores['chain'] = chain_id
                scores['sample'] = sample
                scores['res_idx'] = [i for i in range(len(sampled_seq_i))]
                scores['neg_pll'] = [
                    neg_pll[i].item() for i in range(len(sampled_seq_i))
                ]
                csv_dir = os.path.join(self.save_dir, 'scores')
                os.makedirs(csv_dir, exist_ok=True)
                output_csv(scores, csv_dir)
