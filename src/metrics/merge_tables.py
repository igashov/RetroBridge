import argparse
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs='+')
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    df_list = []
    for csv_file in sorted(args.files, key=str):
        print(csv_file)
        df = pd.read_csv(csv_file)
        df['from_file'] = str(csv_file.name)
        df_list.append(df)

    df_final = pd.concat(df_list)
    df_final.to_csv(args.out, index=False)
