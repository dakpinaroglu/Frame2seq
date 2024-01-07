import os
from glob import glob
from time import time
import torch

from frame2seq.model.Frame2seq import frame2seq
from frame2seq.utils.design import design
from frame2seq.utils.score import score


class Frame2seqRunner():
    """
    Wrapper for Frame2seq predictions.
    """

    def __init__(self):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        module_path = os.path.abspath(__file__)
        globals()['__file__'] = module_path
        project_path = os.path.dirname(os.path.abspath(__file__))
        trained_models_dir = os.path.join(project_path, 'trained_models')

        self.models = []
        model_ckpts = glob(os.path.join(trained_models_dir, '*.ckpt'))
        for ckpt_file in model_ckpts:
            print(f"Loading {ckpt_file}...")
            self.models.append(
                frame2seq.load_from_checkpoint(ckpt_file).eval().to(
                    self.device))

        self.save_dir = os.path.join(os.getcwd(), 'frame2seq_outputs')
        os.makedirs(self.save_dir, exist_ok=True)

    def design(self,
               pdb_file,
               chain_id,
               temperature,
               num_samples,
               omit_AA=None,
               fixed_positions=None,
               save_indiv_seqs=False,
               save_indiv_neg_pll=False,
               verbose=True):
        """
        Design sequences for a given PDB file and chain ID.
        """
        start_time = time()
        design(self, pdb_file, chain_id, temperature, num_samples, omit_AA,
               fixed_positions, save_indiv_seqs, save_indiv_neg_pll, verbose)
        if verbose:
            print(
                f"Designed {num_samples} sequences in {time() - start_time:.2f} seconds."
            )

    def score(self,
              pdb_file,
              chain_id,
              fasta_file=None,
              save_indiv_neg_pll=False,
              verbose=True):
        """
        Score the sequence for a given PDB file and chain ID. Optionally, provide a fasta file with many sequences to score for the given PDB's backbone.
        """
        start_time = time()
        score(self, pdb_file, chain_id, fasta_file, save_indiv_neg_pll)
        if verbose:
            print(f"Scored sequences in {time() - start_time:.2f} seconds.")
