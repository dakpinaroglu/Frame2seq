# Frame2seq
Official repository for Frame2seq, a structured-conditioned masked language model for protein sequence design, as described in our preprint [Structure-conditioned masked language models for protein sequence design generalize beyond the native sequence space](https://doi.org/10.1101/2023.12.15.571823).

## Colab notebook
Colab notebook for generating sequences with Frame2seq: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dakpinaroglu/Frame2seq/blob/main/notebooks/Frame2seq.ipynb)

## Setup
To use Frame2seq, install via pip:
```bash
pip install frame2seq
```

Alternatively, you can clone this repository and install the package locally:
```bash
$ git clone git@github.com:dakpinaroglu/Frame2seq.git
$ pip install Frame2seq
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
- `save_neg_pll`: Whether to save the per-residue negative log-likelihoods of the sampled sequences.
- `verbose`: Whether to print the sampled sequences and time taken for sampling.


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