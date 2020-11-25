import pandas as pd
import sys
import logging


def load(train_path, test_path):
    logging.info("Getting train and test data ...")
    try:
        with open(train_path, 'rb') as infile:
            train_df = pd.read_pickle(infile)
        with open(test_path, 'rb') as infile:
            test_df = pd.read_pickle(infile)
    except IOError:
        sys.exit("Could not read data")

    return train_df, test_df
