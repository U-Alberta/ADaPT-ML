import logging

import numpy as np
import pandas as pd
from kneed import KneeLocator
from pandas.core.common import flatten
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import filter_unlabeled_dataframe
# from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds

from label import (LABEL_MODEL_FILENAME, TRAIN_PARAMS, SQL_QUERY, CRATE_DB_IP)

# from snorkel.labeling.apply.dask import PandasParallelLFApplier

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


def load_lf_info(id_df, features):
    data_df = pd.DataFrame()
    for table in id_df.table.unique():
        table_df = id_df.loc[(id_df.table == table)]
        lf_features_df = pd.read_sql(SQL_QUERY.format(column=', '.join(['id'] + list(features)),
                                                      table=table,
                                                      ids=str(tuple(table_df.id.tolist()))),
                                     CRATE_DB_IP,
                                     chunksize=100)
        for data in lf_features_df:
            data_df = data_df.append(data, ignore_index=True)

    lf_info_df = pd.merge(id_df, data_df, on='id')
    assert id_df.shape[0] == lf_info_df.shape[0]
    # convert to the right data types
    for column in features:
        lf_info_df[column] = lf_info_df[column].apply(lambda d: features[column](d))
    logging.info("LF info loaded. Here's a peek:")
    logging.info(lf_info_df.head())
    return lf_info_df


def create_label_matrix(df, lfs):
    applier = PandasLFApplier(lfs=lfs)
    # applier = PandasParallelLFApplier(lfs=lfs)
    # n_parallel = int(multiprocessing.cpu_count() / 2)
    # train_L = applier.apply(train_df, n_parallel=n_parallel, fault_tolerant=True)
    try:
        label_matrix, metadata = applier.apply(df, return_meta=True)
        if metadata.faults:
            logging.warning("Some LFs failed:", metadata.faults)
    except:
        label_matrix = None
    return label_matrix


def save_label_matrix(label_matrix, filename):
    np.save(filename, label_matrix)


def train_label_model(L_train: np.ndarray, y_dev, labels) -> LabelModel:
    label_model = LabelModel(cardinality=len(labels), **INIT_PARAMS)
    TRAIN_PARAMS.update(LM_OPTIMIZER_SETTINGS[TRAIN_PARAMS['optimizer']])
    label_model.fit(L_train, class_balance=calc_class_balance(y_dev, labels), **TRAIN_PARAMS)
    label_model.save(LABEL_MODEL_FILENAME)
    return label_model


def apply_label_preds(df, label_matrix: np.ndarray, label_model: LabelModel, labels, task) -> pd.DataFrame:
    try:
        filtered_df, probs_array = filter_df(df, label_matrix, label_model)
    except:
        return None
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
    logging.info("Labels inserted. Here's a peek:")
    logging.info(df.head())
    return df


def filter_df(unfiltered_df: pd.DataFrame, label_matrix: np.ndarray, label_model: LabelModel):
    logging.info("Filtering out abstain data points ...")
    try:
        filtered_df, probs = filter_unlabeled_dataframe(unfiltered_df, label_model.predict_proba(label_matrix),
                                                    label_matrix)
        logging.info("data points filtered out: {0}".format(unfiltered_df.shape[0] - filtered_df.shape[0]))
        logging.info("data points remaining: {0}".format(filtered_df.shape[0]))
    except:
        filtered_df = None
        probs = None
    return filtered_df, probs


def validate_training_data(filtered_df, labels):
    logging.info("Validating the training data ...")
    # Make sure each class is represented in the data
    try:
        assert (not set(flatten(filtered_df['label'].tolist())).difference(set([label.name for label in labels])))
    except AssertionError:
        logging.warning("Not all classes are represented in this training data.")


def calc_class_balance(y_dev, labels):
    try:
        y_len = len(y_dev)
        flat_y_dev = np.array(y_dev).flatten().tolist()
        counts = map(lambda l: flat_y_dev.count(l), [label.name for label in labels])
        balance = list(map(lambda count: count / y_len, counts))
    except:
        balance = None
    logging.info("BALANCE: {}".format(balance))
    return balance


def save_df(df, pkl_filename, html_filename):
    df.to_pickle(pkl_filename)
    df.head().to_html(html_filename)
