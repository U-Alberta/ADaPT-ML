import inspect
import sys
import logging

import pandas as pd
import numpy as np
from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier

from label import TRAIN_MATRIX_FILENAME
from label.lfs.keyword import keyword_lfs


def create_df(data_csv_filename: str) -> pd.DataFrame:
    logging.info("Loading data into DataFrame ...")
    return pd.read_csv(data_csv_filename, header=0, index_col=0)


def create_label_matrix(train_df: pd.DataFrame) -> np.ndarray:
    logging.info("Creating label matrix ...")
    applier = PandasLFApplier(lfs=keyword_lfs)
    train_L, metadata = applier.apply(train_df, return_meta=True)
    if metadata.faults:
        logging.warning("Some LFs failed:", metadata.faults)
    return train_L


def save_label_matrix(train_L: np.ndarray):
    np.save(TRAIN_MATRIX_FILENAME, train_L)


def load_label_matrix() -> np.ndarray:
    try:
        return np.load(TRAIN_MATRIX_FILENAME)
    except IOError:
        sys.exit("Label matrix not found. Please run Step 1 first.")
