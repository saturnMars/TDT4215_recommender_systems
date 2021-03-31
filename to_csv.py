#!/usr/bin/env python3
# -*- coding: utf8 -*-

import json
import os

import pandas as pd
from tqdm import tqdm

fields = ['id', 'keywords', 'title', 'createtime']
fields_set = set(fields)


def main(directory, out_file):
    if not out_file:
        print('No output file specified, performing dry-run')
    rows = []
    for file in tqdm(os.listdir(directory)):
        path = os.path.join(directory, file)
        with open(path) as f:
            j = json.load(f)
        rows.append({f['field']: f['value']
                     for f in j['fields']
                     if f['field'] in fields_set})
    df = pd.DataFrame(rows, columns=fields)
    if out_file:
        print('Writing to', out_file)
        df.to_csv(out_file, index=None)
    else:
        print('Dry-run, not saving the following result:')
        print(df)
    print('Success.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input',
                        help='Input directory, should contain 74886 files')
    parser.add_argument('-o', '--output',
                        help='Output file, omit to perform dry-run')

    args = parser.parse_args()

    in_dir = args.input
    out_file = args.output

    main(in_dir, out_file)
