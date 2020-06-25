import inspect

import pandas as pd
import numpy as np
from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier


def create_df(data_csv_filename: str) -> pd.DataFrame:
    print("Loading data into DataFrame ...")
    return pd.read_csv(data_csv_filename, header=0, index_col=0)

def create_label_matrix(train_df: pd.DataFrame) -> np.ndarray:
    applier = PandasLFApplier(lfs=)




def get_lfs():
    lf_list = []
    for thing in inspect.getmembers(lfs):
        if isinstance(thing[1], LabelingFunction):
            lf_list.append(thing[1])
    return lf_list