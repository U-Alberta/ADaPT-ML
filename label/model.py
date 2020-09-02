import logging
import sys

import numpy as np
import pandas as pd
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds

from label.lfs import ValueLabel

INIT_PARAMS = {
    'cardinality': len(ValueLabel),
    'verbose': True,
    'device': 'cpu'
}

LM_OPTIMIZER_SETTINGS = {
    'sgd': {
        'lr': 0.01,
        'l2': 1e-5,                     # keep this between 0 and 0.1
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


def train_label_model(L_train: np.ndarray, label_model_filename: str, train_params: dict) -> LabelModel:
    label_model = LabelModel(**INIT_PARAMS)
    train_params.update(LM_OPTIMIZER_SETTINGS[train_params['optimizer']])
    label_model.fit(L_train, **train_params)
    label_model.save(label_model_filename)
    return label_model


def load_label_model(label_model_filename) -> LabelModel:
    label_model = LabelModel(**INIT_PARAMS)
    try:
        label_model.load(label_model_filename)
        return label_model
    except IOError:
        sys.exit("Label model not found. Please run Step 2 first.")


def apply_label_model(L_train: np.ndarray, label_model: LabelModel, train_df: pd.DataFrame, training_data_filename: str) -> pd.DataFrame:
    probs = label_model.predict_proba(L_train)
    filtered_df, probs = filter_unlabeled_dataframe(train_df, probs, L_train)
    logging.info("data points filtered out: {0}".format(train_df.shape[0]-filtered_df.shape[0]))
    logging.info("data points remaining: {0}".format(filtered_df.shape[0]))
    preds = probs_to_preds(probs)
    pred_labels = [ValueLabel(pred).name if pred != -1 else 'ABSTAIN' for pred in preds]
    filtered_df.insert(len(filtered_df.columns), 'label', pred_labels)
    filtered_df.to_pickle(training_data_filename)
    return filtered_df


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
        sys.exit("Label Model hasn't been trained yet.")
