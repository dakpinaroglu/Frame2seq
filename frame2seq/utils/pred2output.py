import pandas as pd


def output_fasta(model_outs, fasta_dir):
    """
    Given a predicted sequence, write to a fasta file.
    """
    pdbid = model_outs['pdbid']
    chain = model_outs['chain']
    sample = model_outs['sample']
    seq = model_outs['seq']
    recovery = model_outs['recovery']
    avg_neg_pll = model_outs['avg_neg_pll']
    temp = model_outs['temp']

    with open(f"{fasta_dir}/{pdbid}_{chain}_seq{sample}.fasta", "w") as f:
        f.write(
            f">pdbid={pdbid} chain_id={chain} recovery={recovery*100:.2f}% score={avg_neg_pll:.2f} temperature={temp}\n"
        )
        f.write(f"{seq}\n")


def output_csv(scores, csv_dir):
    """
    Given a negative pseudo-log-likelihood, write to a csv file.
    """
    pdbid = scores['pdbid']
    chain = scores['chain']
    sample = scores['sample']
    res_idx = scores['res_idx']
    neg_pll = scores['neg_pll']

    df = pd.DataFrame(
        list(zip(res_idx, neg_pll)),
        columns=['Residue index', 'Negative pseudo-log-likelihood'])
    df.to_csv(f"{csv_dir}/{pdbid}_{chain}_seq{sample}.csv", index=False)
