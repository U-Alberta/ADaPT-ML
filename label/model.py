import logging
import sys
import multiprocessing

import numpy as np
import pandas as pd
from pandas.core.common import flatten
from snorkel.labeling import filter_unlabeled_dataframe
# from snorkel.labeling import PandasLFApplier
from snorkel.labeling.apply.dask import PandasParallelLFApplier

# from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds
from kneed import KneeLocator

from label import (CRATE_DB_IP, TRAIN_DF, LABEL_MATRIX_FILENAME, LABEL_MODEL_FILENAME, TRAINING_DATA_FILENAME,
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

GET_LF_INFO_QUERY = """
    SELECT {column} FROM {table} WHERE id IN {ids};
    """


def load_lf_info(columns):
    try:
        ids = tuple(TRAIN_DF.id.tolist())
        table = TRAIN_DF.at[0, 'table']
        data_df = pd.read_sql(GET_LF_INFO_QUERY.format(column=', '.join(columns), table=table, ids=ids), CRATE_DB_IP)
    except Exception:
        # TODO: we can't do anything about this error so we should just quit probably
        sys.exit("Is CrateDB running and is the data imported?")
    train_df = TRAIN_DF.join(data_df, on='id')
    logging.info("LF info loaded. Here's a peek:")
    logging.info(train_df.head())
    return train_df


def create_label_matrix(train_df, lfs):
    # applier = PandasLFApplier(lfs=lfs)
    applier = PandasParallelLFApplier(lfs=lfs)
    n_parallel = int(multiprocessing.cpu_count() / 2)
    train_L = applier.apply(train_df, n_parallel=n_parallel, fault_tolerant=True)
    # if metadata.faults:
    #     logging.warning("Some LFs failed:", metadata.faults)
    np.save(LABEL_MATRIX_FILENAME, train_L)
    return train_L


def train_label_model(L_train: np.ndarray, labels) -> LabelModel:
    label_model = LabelModel(cardinality=len(labels), **INIT_PARAMS)
    TRAIN_PARAMS.update(LM_OPTIMIZER_SETTINGS[TRAIN_PARAMS['optimizer']])
    label_model.fit(L_train, **TRAIN_PARAMS)
    label_model.save(LABEL_MODEL_FILENAME)
    return label_model


def apply_label_preds(L_train: np.ndarray, label_model: LabelModel, labels, task) -> pd.DataFrame:
    filtered_df, probs_array = filter_df(L_train, label_model)
    probs_list = probs_array.tolist()
    preds_list = probs_to_preds(probs_array).tolist()
    if task == 'multiclass':
        logging.info("Applying Label Model as multiclass ...")
        pred_labels = [[labels(pred).name] for pred in preds_list]
    else:
        logging.info("Applying Label Model as multilabel ...")
        label_values = [label.value for label in labels]
        pred_labels = []
        for probs in probs_list:
            pairs = list(zip(label_values, probs))
            pairs.sort(key=lambda t: t[1])
            kneedle = KneeLocator(label_values,
                                  [pair[1] for pair in pairs],
                                  S=1.0,
                                  curve="convex",
                                  direction="increasing",
                                  online=True)
            try:
                assert kneedle.knee_y is not None
                chosen = [labels(pair[0]).name for pair in pairs if pair[1] > kneedle.knee_y]
            except AssertionError:
                chosen = [labels(pairs[-1][0]).name]
            pred_labels.append(chosen)
    return add_labels(filtered_df, pred_labels, probs_list)


def add_labels(df, labels, label_probs):
    df.insert(len(df.columns), 'label', labels)
    df.insert(len(df.columns), 'label_probs', label_probs)
    df.to_pickle(TRAINING_DATA_FILENAME)
    df.head().to_html(TRAINING_DATA_HTML_FILENAME)
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
