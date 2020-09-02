import logging
import sys

import numpy as np
import pandas as pd
from snorkel.labeling import PandasLFApplier

from label.lfs.pv import keyword_lfs


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
