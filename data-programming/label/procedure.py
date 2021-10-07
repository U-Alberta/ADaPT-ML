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

from label import TRAIN_PARAMS, SQL_QUERY, DATABASE_IP

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
        if features[column] is not None:
            lf_info_df[column] = lf_info_df[column].apply(lambda d: features[column](d))
    logging.info("LF info loaded. Here's a peek:")
    logging.info(lf_info_df.head())
    return lf_info_df


def create_label_matrix(df, lfs, parallel) -> np.ndarray:
    """
    This function applies the LFs over the datapoints in the given DataFrame to create an n X m numpy matrix where each
    row is for a datapoint and each column is the integer vote from each LF if its check is successful or -1 if it
    abstains.
    :param df: pd.DataFrame containing columns with extracted features from the datapoint necessary for the current
    classification task's LFs
    :param lfs: The current classification task's [LabelingFunction]
    :param parallel: Choose whether to apply the labeling functions in sequence or in parallel
    :return: numpy.ndarray of integers representing votes from the LFs
    """
    if parallel:
        applier = PandasParallelLFApplier(lfs=lfs)
        n_parallel = int(multiprocessing.cpu_count() / 2)
        label_matrix = applier.apply(df, n_parallel=n_parallel, fault_tolerant=False)
    else:
        applier = PandasLFApplier(lfs=lfs)
        label_matrix = applier.apply(df)
    return label_matrix


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
    if TRAIN_PARAMS['seed'] is None:
        TRAIN_PARAMS['seed'] = np.random.randint(1e6)
    try:
        balance = calc_class_balance(y_dev, labels)
        logging.info("TRUE CLASS BALANCE: {}".format(balance))
        balance = list(balance.values())
    except:
        balance = None
        logging.info("TRUE CLASS BALANCE: {}".format(balance))
    label_model.fit(L_train, class_balance=balance, **TRAIN_PARAMS)
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
    # filter out abstain datapoints
    filtered_df, probs_array = filter_df(df, label_matrix, label_model)

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
            chosen = find_knee(labels, label_values, probs)
            pred_labels.append(chosen)
    return add_labels(filtered_df, pred_labels, probs_list)


def find_knee(labels, label_values, probs) -> [str]:
    """
    This function is used in a multilabel setting to use the label model's probabilities to find which labels should
    be included in the prediction. It locates probabilities above the knee and adds them to the list.
    Reference: https://towardsdatascience.com/using-snorkel-for-multi-label-annotation-cc2aa217986a

    :param labels: the Label class for the current classification task that stores the name and int value
    :label values: a list of the labels' int values in order that they appear with their probabilities
    :probs: a list of probabilities for the labels from the label model that sum to 1
    :return: a list of labels with probabilities above the knee, or the most probable label if no knee is detected
    """
    # get a list of tuples with the label's int value followed by its probability, then sort from lowest to
    # highest probability
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


def add_labels(df: pd.DataFrame, labels: [[str]], label_probs: [[float]]) -> pd.DataFrame:
    """
    This function adds the list of labels and label probabilities to the dataframe
    :param df: pd.DataFrame that is unlabeled
    :param labels: The [[Label.name]] for the label model's predictions for these datapoints
    :param label_probs: The probabilities associated with all labels, not just the chosen ones
    :return: pd.DataFrame that has columns id, table, label, label_probs
    """
    df.insert(len(df.columns), 'label', labels)
    df.insert(len(df.columns), 'label_probs', label_probs)
    logging.info("Labels inserted. Here's a peek:")
    logging.info(df.head())
    return df


def filter_df(df: pd.DataFrame, label_matrix: np.ndarray, label_model: LabelModel) -> (pd.DataFrame, [[float]]):
    """
    This function uses the label model's predictions to filter out data points that cannot be classified.
    :param df: a pd.DataFrame containing data points to be filtered
    :param label_matrix: The numpy.ndarray of LF votes for each data point
    :param label_model: The Label Model that has been trained on the train label matrix
    :return: The filtered df and corresponding probabilities
    """
    logging.info("Filtering out abstain data points ...")
    filtered_df, probs = filter_unlabeled_dataframe(df, label_model.predict_proba(label_matrix),
                                                    label_matrix)
    logging.info("data points filtered out: {0}".format(df.shape[0] - filtered_df.shape[0]))
    logging.info("data points remaining: {0}".format(filtered_df.shape[0]))
    return filtered_df, probs


def validate_training_data(df: pd.DataFrame, labels):
    """
    - Do a check of the class representation for the training data
    :param df: The labeled train pd.DataFrame
    :param labels: the Label class for this current classification task storing the name and int value for each class
    """
    logging.info("Validating the training data ...")
    # Check class balance
    balance = calc_class_balance(df.label.tolist(), labels)
    logging.info("TRAINING DATA CLASS BALANCE: {}".format(balance))


def calc_class_balance(y: [[str]], labels) -> dict:
    """
    Calculates class balance for multilabel and multiclass annotations by flattening the multidimensional array of
    labels into one array, counting the number of times each label appears, and dividing that count by the total number
    of data points to get each label's proportion of the entire dataset. We take the proportion out of data points, not
    the total number of labels, because we assume that in a multilabel setting, each label per data point can fully
    describe that data point without depending on the other label to complete it.
            Example:
                y = [['cat'],['dog','bird'],['cat','dog']]
                label_names = ['cat','dog','bird']
                y_len = 3
                counts = [2, 2, 1]
                balance = {'cat':0.66, 'dog':0.66, 'bird':0.33}
    :param y: The annotations
    """
    y_len = len(y)
    flat_y_dev = [label for sublist in y for label in sublist]
    label_names = [label.name for label in labels]
    counts = map(lambda l: flat_y_dev.count(l), label_names)
    balance = list(map(lambda count: count / y_len, counts))
    return dict(zip(label_names, balance))


def save_df(df: pd.DataFrame, pkl_filename: str, html_filename: str):
    logging.info("Here's a preview of {}:".format(pkl_filename))
    logging.info(df.head())
    df.to_pickle(pkl_filename)
    df.head().to_html(html_filename)


def save_label_matrix(label_matrix: np.ndarray, filename: str):
    np.save(filename, label_matrix)
