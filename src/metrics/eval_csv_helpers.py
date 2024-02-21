import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
from rdkit import Chem


def canonicalize(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return np.nan
    return Chem.MolToSmiles(m)


def _assign_groups(df, samples_per_product):
    df['group'] = np.arange(len(df)) // samples_per_product
    return df


def assign_groups(df, samples_per_product_per_file=10):
    df = df.groupby('from_file').apply(partial(_assign_groups, samples_per_product=samples_per_product_per_file))
    return df


def compute_confidence(df):
    counts = df.groupby(['group', 'pred']).size().reset_index(name='count')
    group_size = df.groupby(['group']).size().reset_index(name='group_size')

    #     # Don't use .merge() as it can change the order of rows
    # #     df = df.merge(counts, on=['group', 'pred'], how='left')
    #     df = df.merge(counts, on=['group', 'pred'], how='inner')
    #     df = df.merge(group_size, on=['group'], how='left')

    counts_dict = {(g, p): c for g, p, c in zip(counts['group'], counts['pred'], counts['count'])}
    df['count'] = df.apply(lambda x: counts_dict[(x['group'], x['pred'])], axis=1)

    size_dict = {g: s for g, s in zip(group_size['group'], group_size['group_size'])}
    df['group_size'] = df.apply(lambda x: size_dict[x['group']], axis=1)

    df['confidence'] = df['count'] / df['group_size']

    # sanity check
    assert (df.groupby(['group', 'pred'])['confidence'].nunique() == 1).all()
    assert (df.groupby(['group'])['group_size'].nunique() == 1).all()

    return df


def get_top_k(df, k, scoring=None):
    if callable(scoring):
        df["_new_score"] = scoring(df)
        scoring = "_new_score"

    if scoring is not None:
        df = df.sort_values(by=scoring, ascending=False)
    df = df.drop_duplicates(subset='pred')

    return df.head(k)


def compute_accuracy(df, top=[1, 3, 5], scoring=None, verbose=False):
    round_trip = 'pred_product' in df.columns

    results = {}
    results['Exact match'] = {}

    df['exact_match'] = df['true'] == df['pred']

    if round_trip:
        #         results['Round-trip only coverage'] = {}
        results['Round-trip coverage'] = {}
        #         results['Round-trip only accuracy'] = {}
        results['Round-trip accuracy'] = {}

        df['round_trip_match'] = df['product'] == df['pred_product']
        df['match'] = df['exact_match'] | df['round_trip_match']

    for k in tqdm(top):
        topk_df = df.groupby(['group']).apply(partial(get_top_k, k=k, scoring=scoring)).reset_index(drop=True)

        acc_exact_match = topk_df.groupby('group').exact_match.any().mean()
        results['Exact match'][f'top-{k}'] = acc_exact_match
        if verbose:
            print(f"\nTop-{k}")
            print("Exact match accuracy", acc_exact_match)

        if round_trip:
            #             cov_round_trip_only = topk_df.groupby('group').round_trip_match.any().mean()
            cov_round_trip = topk_df.groupby('group').match.any().mean()
            #             acc_round_trip_only = topk_df.groupby('group').round_trip_match.mean().mean()
            acc_round_trip = topk_df.groupby('group').match.mean().mean()
            #             acc_round_trip_only = topk_df.round_trip_match.mean()
            #             acc_round_trip = topk_df.match.mean()

            #             results['Round-trip only coverage'][f'top-{k}'] = cov_round_trip_only
            results['Round-trip coverage'][f'top-{k}'] = cov_round_trip
            #             results['Round-trip only accuracy'][f'top-{k}'] = acc_round_trip_only
            results['Round-trip accuracy'][f'top-{k}'] = acc_round_trip

            if verbose:
                #                 print("Round-trip only coverage", cov_round_trip_only)
                print("Round-trip coverage", cov_round_trip)
                #                 print("Round-trip only accuracy", acc_round_trip_only)
                print("Round-trip accuracy", acc_round_trip)

    return pd.DataFrame(results).T
