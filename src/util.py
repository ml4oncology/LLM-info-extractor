"""Utility module
"""
import pandas as pd

###############################################################################
# I/O
###############################################################################
def load_data(data_path: str) -> pd.DataFrame:
    if data_path.endswith('.parquet') or data_path.endswith('.parquet.gzip'):
        df = pd.read_parquet(data_path)
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    return df


def save_data(df: pd.DataFrame, save_path: str, **kwargs):
    if save_path.endswith('.parquet'):
        df.to_parquet(save_path, **kwargs)
    elif save_path.endswith('.parquet.gzip'):
        df.to_parquet(save_path, compression='gzip', **kwargs)
    elif save_path.endswith('.csv'):
        df.to_csv(save_path, **kwargs)
    elif save_path.endswith('.xlsx'):
        df.to_excel(save_path, **kwargs)