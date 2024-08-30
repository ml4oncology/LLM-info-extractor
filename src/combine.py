"""
Script to combine the partitioned results
"""
import argparse
import glob
import json

import pandas as pd

from ml_common.util import load_table, save_table


def fix_failed_output(df: pd.DataFrame):
    if 'failed_output' not in df:
        return df
    
    mask = df['failed_output'].notnull()
    fixed_output = []
    for generated_text in df.loc[mask, 'failed_output']:
        if generated_text.endswith('"'):
            # reached max token 
            generated_text += '}'
        
        if not generated_text.endswith('"}'):
            # reached max token 
            generated_text += '"}'

        # replace double quotes in reasoning with single quote
        start_idx = generated_text.index('"Reason": "') + len('"Reason": "')
        end_idx = -3
        generated_text = (
            generated_text[:start_idx] 
            + generated_text[start_idx:end_idx].replace('"', '\'') 
            + generated_text[end_idx:]
        )

        try:
            result = json.loads(generated_text)
        except json.JSONDecodeError:
            result = {'failed_output': generated_text}
        fixed_output.append(result)
    
    fixed_output = pd.DataFrame(fixed_output, index=df.index[mask])
    df.loc[mask, fixed_output.columns] = fixed_output
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--partition-filepath', 
        type=str, 
        help=(
            "Path to files to be merged; enclose in quotes, accepts * as wildcard for directories or filenames "
            "(e.g. './data_partitions/prompted_*_pe_thorax_ct.xlsx)'"
        )
    )
    parser.add_argument('--output-filepath', type=str, help='Where to output the combined result')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    res = pd.concat([load_table(path) for path in glob.glob(args.partition_filepath)])
    res = fix_failed_output(res)
    res = res.sort_values(by='index')
    save_table(res, args.output_filepath, index=False)
    
if __name__ == '__main__':
    main()
