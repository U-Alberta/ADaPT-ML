import logging
import sys

import numpy as np
import pandas as pd
from pandas.core.common import flatten
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling import PandasLFApplier
# from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds
from kneed import KneeLocator

from label import (TRAIN_DF, LABEL_MATRIX_FILENAME, LABEL_MODEL_FILENAME, TRAINING_DATA_FILENAME,
                   TRAINING_DATA_HTML_FILENAME, TRAIN_PARAMS)

INIT_PARAMS = {
    'verbose': True,
    'device': 'cpu'
}

LM_OPTIMIZER_SETTINGS = {
    'sgd': {
        'lr': 0.01,
        'l2': 1e-5,  # keep this between 0 and 0.1
        'lr_scheduler': 'constant'
    },
    'adam': {
        'lr': 0.001,
        'l2': 1e-5,
        'lr_scheduler': 'constant'
    },
    'adamax': {
        'lr': 0.002,
        'l2': 1e-5,
        'lr_scheduler': 'constant'
    }
}


def create_label_matrix(lfs):
    logging.info("Creating label matrix ...")
    applier = PandasLFApplier(lfs=lfs)
    train_L, metadata = applier.apply(TRAIN_DF, return_meta=True)
    if metadata.faults:
        logging.warning("Some LFs failed:", metadata.faults)
    np.save(LABEL_MATRIX_FILENAME, train_L)
    return train_L


def train_label_model(L_train: np.ndarray, labels) -> LabelModel:
    label_model = LabelModel(cardinality=len(labels), **INIT_PARAMS)
    TRAIN_PARAMS.update(LM_OPTIMIZER_SETTINGS[TRAIN_PARAMS['optimizer']])
    label_model.fit(L_train, **TRAIN_PARAMS)
    label_model.save(LABEL_MODEL_FILENAME)
    return label_model


def apply_multiclass(L_train: np.ndarray, label_model: LabelModel, labels) -> pd.DataFrame:
    logging.info("Applying Label Model as multiclass ...")
    filtered_df, probs = filter_df(L_train, label_model)
    return add_labels(filtered_df, [labels(pred).name for pred in probs_to_preds(probs).tolist()], probs.tolist())


def apply_multilabel(L_train: np.ndarray, label_model: LabelModel, labels) -> pd.DataFrame:
    logging.info("Applying Label Model as multilabel ...")
    filtered_df, probs_array = filter_df(L_train, label_model)
    probs_list = probs_array.tolist()
    label_values = [label.value for label in labels]
    multilabels = []
    for probs in probs_list:
        pairs = list(zip(label_values, probs))
        pairs.sort(key=lambda t: t[1])
        kneedle = KneeLocator([pair[0] for pair in pairs],
                              [pair[1] for pair in pairs],
                              S=1.0,
                              curve="convex",
                              direction="increasing")
        try:
            assert kneedle.knee_y is not None
            chosen = [labels(pair[0]).name for pair in pairs if pair[1] > kneedle.knee_y]
        except AssertionError:
            chosen = [labels(pairs[-1][0]).name]
        multilabels.append(chosen)
    return add_labels(filtered_df, multilabels, probs_list)


def add_labels(df, labels, label_probs):
    df.insert(len(df.columns), 'label', labels)
    df.insert(len(df.columns), 'label_probs', label_probs)
    df.to_pickle(TRAINING_DATA_FILENAME)
    df.to_html(TRAINING_DATA_HTML_FILENAME)
    return df


def filter_df(L_train: np.ndarray, label_model: LabelModel):
    logging.info("Filtering out abstain data points ...")
    filtered_df, probs = filter_unlabeled_dataframe(TRAIN_DF, label_model.predict_proba(L_train), L_train)
    logging.info("data points filtered out: {0}".format(TRAIN_DF.shape[0] - filtered_df.shape[0]))
    logging.info("data points remaining: {0}".format(filtered_df.shape[0]))
    return filtered_df, probs


def validate_training_data(filtered_df, labels):
    logging.info("Validating the training data ...")
    # Make sure each class is represented in the data
    try:
        assert (not set(flatten(filtered_df['label'].tolist())).difference(set([label.name for label in labels])))
    except AssertionError:
        logging.warning("Not all classes are represented in this training data.")


def train_params_dict(label_model: LabelModel) -> dict:
    try:
        train_params = {
            'n_epochs': label_model.train_config.n_epochs,
            'optimizer': label_model.train_config.optimizer,
            'lr_scheduler': label_model.train_config.lr_scheduler,
            'lr': label_model.train_config.lr,
            'l2': label_model.train_config.l2,
            'prec_init': label_model.train_config.prec_init
        }
        return train_params
    except AttributeError:
        sys.exit("Label Model hasn't been trained yet...?")
