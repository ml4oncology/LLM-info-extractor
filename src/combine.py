"""
Script to combine the partitioned results
"""
import argparse
import glob

import pandas as pd

from llm_info_extractor.util import load_data, save_data

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
    res = pd.concat([load_data(path) for path in glob.glob(args.partition_filepath)])
    res = res.sort_values(by='index')
    save_data(res, args.output_filepath)
    
if __name__ == '__main__':
    main()
