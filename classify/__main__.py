from classify import (parsed_args, bundle, BUNDLED_FILENAME)
from label import TRAINING_DATA_FILENAME, DEMO_TRAINING_DATA_FILENAME
import pandas as pd
import logging
import sys


if __name__ == '__main__':

    logging.info("Loading training data ...")
    try:
        train_df = pd.read_pickle(DEMO_TRAINING_DATA_FILENAME)
    except IOError:
        sys.exit("Training data does not exist. Please run label first.")
    if parsed_args.model == 0:
        model_bundle = bundle.MlogitModelBundle()
        model_bundle.train(train_df.all_text.tolist(), train_df.label.tolist())
        model_bundle.save(BUNDLED_FILENAME)
