import argparse
import re
from pathlib import Path
import sys
import pandas as pd
from time import time

"""
Chen, Shuan, and Yousung Jung.
"Deep retrosynthetic reaction prediction using local reactivity and global attention."
JACS Au 1.10 (2021): 1612-1620.

Predicted precursors are considered correct if 
- the predicted precursors are the same as the ground truth
- Molecular Transformer predicts the target product for the proposed precursors
"""


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    https://github.com/pschwllr/MolecularTransformer/tree/master#pre-processing
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens)
    if smi != ''.join(tokens):
        print(smi, ''.join(tokens))
    return ' '.join(tokens)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=Path, required=True)
    parser.add_argument("--csv_out", type=Path, required=False)
    parser.add_argument("--mol_trans_dir", type=Path, default="./MolecularTransformer/")
    args = parser.parse_args()

    sys.path.append(str(args.mol_trans_dir))
    import onmt
    from onmt.translate.translator import build_translator
    import onmt.opts

    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    args = parser.parse_args(sys.argv[1:] + [
        "-model", str(Path(args.mol_trans_dir, 'models', 'MIT_mixed_augm_model_average_20.pt')),
        "-src", "input.txt", "-output", "pred.txt ",
        "-replace_unk", "-max_length", "200", "-fast"
    ])

    # Read CSV
    df = pd.read_csv(args.csv_file)

    # Find unique SMILES
    unique_smiles = list(set(df['pred']))

    # Tokenize
    tokenized_smiles = [smi_tokenizer(s.strip()) for s in unique_smiles]

    print("Predicting products...")
    tic = time()
    translator = build_translator(args, report_score=True)
    scores, pred_products = translator.translate(
        src_data_iter=tokenized_smiles,
        batch_size=args.batch_size,
        attn_debug=args.attn_debug
    )
    pred_products = [x[0].strip() for x in pred_products]
    print("... done after {} seconds".format(time() - tic))

    # De-tokenize
    pred_products = [''.join(x.split()) for x in pred_products]

    # gather results
    pred_products = {r: p for r, p in zip(unique_smiles, pred_products)}

    # update dataframe
    df['pred_product'] = [pred_products[r] for r in df['pred']]

    # Write results
    if args.csv_out:
        print("Writing CSV file...")
        df.to_csv(args.csv_out, index=False)
