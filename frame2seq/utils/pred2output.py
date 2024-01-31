import pandas as pd


def output_fasta(preds, fasta_dir):
    """
    Given predicted sequences, write to a fasta file.
    """
    with open(f"{fasta_dir}/seqs.fasta", "a") as f:
        for sample_i in range(len(preds)):
            pdbid_i = preds[sample_i][0]
            chain_i = preds[sample_i][1]
            seq_i = preds[sample_i][3]
            recovery_i = preds[sample_i][4]
            avg_neg_pll_i = preds[sample_i][5]
            temp_i = preds[sample_i][6]
            f.write(
                f">pdbid={pdbid_i} chain_id={chain_i} recovery={recovery_i*100:.2f}% score={avg_neg_pll_i:.2f} temperature={temp_i}\n"
            )
            f.write(f"{seq_i}\n")


def output_indiv_fasta(preds, fasta_dir):
    """
    Given a predicted sequence, write to a fasta file.
    """
    pdbid = preds[0]
    chain = preds[1]
    sample = preds[2]
    seq = preds[3]
    recovery = preds[4]
    avg_neg_pll = preds[5]
    temp = preds[6]

    with open(f"{fasta_dir}/{pdbid}_{chain}_seq{sample}.fasta", "w") as f:
        f.write(
            f">pdbid={pdbid} chain_id={chain} recovery={recovery*100:.2f}% score={avg_neg_pll:.2f} temperature={temp}\n"
        )
        f.write(f"{seq}\n")


def output_csv(preds, csv_dir):
    """
    Given average negative pseudo-log-likelihoods, write to a csv file.
    """
    df = pd.DataFrame(columns=[
        'PDBID', 'Chain ID', 'Sample Number', 'Scored sequence',
        'Average negative pseudo-log-likelihood', 'Temperature'
    ],
                      data=preds)
    df.to_csv(f"{csv_dir}/scores.csv", index=False)


def output_indiv_csv(scores, csv_dir):
    """
    Given per-residue negative pseudo-log-likelihoods, write to a csv file.
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
