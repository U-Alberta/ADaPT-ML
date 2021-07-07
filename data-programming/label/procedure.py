import logging
import multiprocessing

import numpy as np
import pandas as pd
from kneed import KneeLocator
from pandas.core.common import flatten
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import filter_unlabeled_dataframe
# from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds

from label import (LABEL_MODEL_FILENAME, TRAIN_PARAMS, SQL_QUERY, DATABASE_IP)

from snorkel.labeling.apply.dask import PandasParallelLFApplier

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


def load_lf_info(id_df, features) -> pd.DataFrame:
    """
    This function takes a DataFrame containing two columns, table and id, of all of the data points that have been
    selected through some means to become the labelled training data. It looks up the features needed for the LFs and
    adds this to the DataFrame.
    :param id_df: pd.DataFrame with columns table and id
    :param features: dict with column name key and function value that determines which LF features are needed and how
    to process them from the database
    :return: pd.DataFrame of id, table, and columns with features extracted from the datapoints that are necessary
    for the current classification task's LFs
    """
    data_df = pd.DataFrame()
    # get the feature columns from the database one table at a time
    for table in id_df.table.unique():
        table_df = id_df.loc[(id_df.table == table)]
        lf_features_df = pd.read_sql(SQL_QUERY.format(column=', '.join(['id'] + list(features)),
                                                      table=table,
                                                      ids=str(tuple(table_df.id.tolist()))),
                                     DATABASE_IP,
                                     chunksize=100)
        for data in lf_features_df:
            data_df = data_df.append(data, ignore_index=True)

    # join the dataframes so that the collected features line up with the ids.
    lf_info_df = pd.merge(id_df, data_df, on='id')
    assert id_df.shape[0] == lf_info_df.shape[0]

    # convert to the right data types
    for column in features:
        lf_info_df[column] = lf_info_df[column].apply(lambda d: features[column](d))
    logging.info("LF info loaded. Here's a peek:")
    logging.info(lf_info_df.head())
    return lf_info_df


def create_label_matrix(df, lfs):
    """
    This function applies the LFs over the datapoints in the given DataFrame to create an n X m numpy matrix where each
    row is for a datapoint and each column is the integer vote from each LF if its check is successful or -1 if it
    abstains.
    :param df: pd.DataFrame containing columns with extracted features from the datapoint necessary for the current
    classification task's LFs
    :param lfs: The current classification task's [LabelingFunction]
    :return: numpy.ndarray of integers representing votes from the LFs
    """
    applier = PandasParallelLFApplier(lfs=lfs)
    n_parallel = int(multiprocessing.cpu_count() / 2)
    try:
        label_matrix = applier.apply(df, n_parallel=n_parallel, fault_tolerant=False)
    except:
        label_matrix = None
    return label_matrix


def save_label_matrix(label_matrix, filename):
    np.save(filename, label_matrix)


def train_label_model(L_train: np.ndarray, y_dev: [[str]], labels) -> LabelModel:
    """
    Uses the label matrix as input to train the Label Model to estimate the true label for each datapoint by learning
    the correlations, conflicts, and general noise of the LFs. It optionally calculates the class balance using the
    development labels from Label Studio if they are available.
    :param L_train: the label matrix created from the train DataFrame
    :param y_dev: the list of labels from Label Studio for each datapoint
    :param labels: The Label class for the current classification task that holds the name and int value of each class
    :return: The trained Label Model
    """
    label_model = LabelModel(cardinality=len(labels), **INIT_PARAMS)
    # to change the optimizer parameters, refer to LM_OPTIMIZER_SETTINGS at the top of this module
    TRAIN_PARAMS.update(LM_OPTIMIZER_SETTINGS[TRAIN_PARAMS['optimizer']])
    label_model.fit(L_train, class_balance=calc_class_balance(y_dev, labels), **TRAIN_PARAMS)
    label_model.save(LABEL_MODEL_FILENAME)
    return label_model


def apply_label_preds(df, label_matrix: np.ndarray, label_model: LabelModel, labels, task) -> pd.DataFrame:
    """
    This function behaves differently depending on whether the task is in a multiclass or multilabel setting. First,
    datapoints in the train DataFrame that were not labeled by the Label Model are filtered out. If the setting is
    multiclass, the predictions and probabilities are added to the DataFrame and the data is returned. If the setting
    is multilabel, the probabilities are sorted and the labels with the probs higher than the knee are taken as the
    predictions.
    :param df: the pd.DataFrame that has a corresponding label matrix that the label model will make predictions for
    :param label_matrix: the label matrix that belongs to the df
    :param label_model: the label model that has been trained on the train label matrix
    :param labels: the Label class for this current classification task storing the name and int value for each class
    :param task: one of multilabel or multiclass
    :return: a pd.DataFrame containing the columns id, table, label, label_probs
    """
    try:
        # filter out abstain datapoints
        filtered_df, probs_array = filter_df(df, label_matrix, label_model)
    except:
        return None
    # get the list of labels predicted to be true and probabilities corresponding to each possible label
    probs_list = probs_array.tolist()
    preds_list = probs_to_preds(probs_array).tolist()
    if task == 'multiclass':
        logging.info("Applying Label Model as multiclass ...")
        # convert predicted labels from their int value to their name
        pred_labels = [[labels(pred).name] for pred in preds_list]
    else:
        logging.info("Applying Label Model as multilabel ...")
        label_values = [label.value for label in labels]
        pred_labels = []
        for probs in probs_list:
            # get a list of tuples with the label's int value followed by its probability, then sort from lowest to
            # highest probability
            chosen = find_knee(labels, label_values, probs)
            pred_labels.append(chosen)
    return add_labels(filtered_df, pred_labels, probs_list)


def find_knee(labels, label_values, probs)
    pairs = list(zip(label_values, probs))
    pairs.sort(key=lambda t: t[1])
    # find the point of sharpest increase in probability
    kneedle = KneeLocator(label_values,
                          [pair[1] for pair in pairs],
                          S=1.0,
                          curve="convex",
                          direction="increasing",
                          online=True)
    try:
        assert kneedle.knee_y is not None
        # add label names to the prediction list for labels that have a probability that is larger than the knee
        chosen = [labels(pair[0]).name for pair in pairs if pair[1] > kneedle.knee_y]
    except AssertionError:
        # if no knee is found, add the label name with the highest probability
        chosen = [labels(pairs[-1][0]).name]
    return chosen


def add_labels(df, labels, label_probs):
    """
    This function adds the list of labels and
    """
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
        assert 0 not in balance
    except:
        balance = None
    logging.info("BALANCE: {}".format(balance))
    return balance


def save_df(df, pkl_filename, html_filename):
    df.to_pickle(pkl_filename)
    df.head().to_html(html_filename)
