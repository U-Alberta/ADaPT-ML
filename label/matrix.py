import inspect
import sys
import logging

import pandas as pd
import numpy as np
from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier

from label import DEMO_CSV_FILENAME, DEMO_DF_FILENAME, DEMO_MATRIX_FILENAME
from label.lfs.keyword import keyword_lfs, evaluate_lfs


def create_df(data_csv_filename: str, train_df_filename: str) -> pd.DataFrame:
    logging.info("Loading data into DataFrame ...")
    try:
        return pd.read_pickle(train_df_filename)
    except IOError:
        train_df = pd.read_csv(data_csv_filename, header=0, index_col=0)
        train_df.to_pickle(train_df_filename)
        return train_df


def create_label_matrix(train_df: pd.DataFrame, matrix_filename: str) -> np.ndarray:
    logging.info("Creating label matrix ...")
    applier = PandasLFApplier(lfs=keyword_lfs)
    train_L, metadata = applier.apply(train_df, return_meta=True)
    if metadata.faults:
        logging.warning("Some LFs failed:", metadata.faults)
    np.save(matrix_filename, train_L)
    return train_L


def load_label_matrix(matrix_filename: str) -> np.ndarray:
    try:
        return np.load(matrix_filename)
    except IOError:
        sys.exit("Label matrix not found. Please run Step 1 first.")


if __name__ == '__main__':
    demo_df = create_df(DEMO_CSV_FILENAME, DEMO_DF_FILENAME)
    demo_matrix = create_label_matrix(demo_df, DEMO_MATRIX_FILENAME)
    with pd.option_context('display.max_colwidth', -1):
        print("Here's a peek at the demo DF:\n", demo_df.head())
        print("Here's a peek at the demo label matrix:\n", demo_matrix[:5, :])
    lf_summary = evaluate_lfs(demo_matrix)
    print("Here's a summary of the LFs:\n", lf_summary)
