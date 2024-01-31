# Frame2seq
Official repository for Frame2seq, a structured-conditioned masked language model for protein sequence design, as described in our preprint [Structure-conditioned masked language models for protein sequence design generalize beyond the native sequence space](https://doi.org/10.1101/2023.12.15.571823).

<p align="center"><img src="https://github.com/dakpinaroglu/Frame2seq/blob/main/.github/frame2seq_net_arc.png"/></p>

## Colab notebook
Colab notebook for generating sequences with Frame2seq: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dakpinaroglu/Frame2seq/blob/main/Frame2seq.ipynb)

## Setup
To use Frame2seq, install via pip:
```bash
pip install frame2seq
```
If previously installed via pip, upgrade to the latest version:
```bash
pip install --upgrade frame2seq
```

## Usage

### Sequence design

To use Frame2seq to generate sequences, you can use the `design` function.

```python
from frame2seq import Frame2seqRunner

runner = Frame2seqRunner()
runner.design(pdb_file, chain_id, temperature, num_samples)
```

#### Arguments

- `pdb_file`: Path to PDB file.
- `chain_id`: Chain ID of protein.
- `temperature`: Sampling temperature.
- `num_samples`: Number of sequences to sample.
- `omit_AA`: Amino acids to omit from sampling. Pass a list of single-letter amino acid strings (e.g. `['C', 'M']`). Default None.
- `fixed_positions`: Amino acid positions to fix during sampling. Pass a list of integers (e.g. `[1, 3, 11]`). Residue numbering starts from 1. Default None.
- `save_indiv_seqs`: Whether to save sequences to indidual .fasta files. Default False.
- `save_indiv_neg_pll`: Whether to save the per-residue negative pseudo-log-likelihoods of the sampled sequences. Default False.
- `verbose`: Whether to print the sampled sequences and time taken for sampling. Default True.

#### Outputs 

A .fasta file containing all sampled sequences is automatically saved. If `save_indiv_seqs` is True, individual .fasta files for each sampled sequence are also saved. 

```
>pdbid=2fra chain_id=A recovery=62.67% score=0.83 temperature=1.0
PPSSVDWRDLGCITDVLDMGGCGACWAFSAVGALEARTTQKTGELTRLSAQDLVDCAREKYGNEGCDGGRMKSSFQFIIDKNGIDSHQAYPFTASDQECLYNSKYKAATCTDYTVLPEGDEDKLREAVSNVGPVAVGIDATHPEFRNFKSGVYHDPKCTTETNHGVLVVGYGTLKGKRFYKVKTCWGTYFGEDGFIRVAKNQGNHCGISTDPSYPEM
```

If `save_indiv_neg_pll` is True, a .csv file containing the per-residue negative pseudo-log-likelihoods of the sampled sequences is also saved.

### Advanced sequence design

To use Frame2seq to generate sequences with advanced options, you can use the `design` function with additional arguments.

```python
from frame2seq import Frame2seqRunner

runner = Frame2seqRunner()
runner.design(pdb_file, chain_id, temperature, num_samples, omit_AA=['C'], fixed_positions=[1, 3, 11])
```

### Scoring

To use Frame2seq to score sequences, you can use the `score` function.

The following will score the PDB sequence for the PDB backbone.
```python
from frame2seq import Frame2seqRunner

runner = Frame2seqRunner()
runner.score(pdb_file, chain_id)
```

The following will score all sequences in the given .fasta file for the PDB backbone.
```python
from frame2seq import Frame2seqRunner

runner = Frame2seqRunner()
runner.score(pdb_file, chain_id, fasta_file)
```

#### Arguments

- `pdb_file`: Path to PDB file.
- `chain_id`: Chain ID of protein.
- `fasta_file`: Path to .fasta file containing sequences to score. Default None. If None, will score the PDB sequence.
- `save_indiv_neg_pll`: Whether to save the per-residue negative pseudo-log-likelihoods of the given sequence(s). Default False.
- `verbose`: Whether to print time taken for scoring. Default True.

#### Outputs 

A .csv file containing the average negative pseudo-log-likelihoods of the given sequence(s) is automatically saved. If `save_indiv_neg_pll` is True, per-residue negative pseudo-log-likelihoods are also saved in individual .csv files.


## Citing this work

```bibtex
@article{akpinaroglu2023structure,
  title={Structure-conditioned masked language models for protein sequence design generalize beyond the native sequence space},
  author={Akpinaroglu, Deniz and Seki, Kosuke and Guo, Amy and Zhu, Eleanor and Kelly, Mark JS and Kortemme, Tanja},
  journal={bioRxiv},
  pages={2023--12},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

[![DOI](https://zenodo.org/badge/DOI/10.1101/2023.12.15.571823.svg)](https://doi.org/10.1101/2023.12.15.571823)

[zenodo](https://zenodo.org/records/10431300)
