import os.path

import pandas as pd
import numpy as np
import sys
import os
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from model_objs import SQL_QUERY, DATABASE_IP


def load(train_path: str, test_path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads the training and testing pd.DataFrames from disk
    :param train_path: path to the training data given as an argument on startup
    :param test_path: path to the test data given as an argument on startup
    :return: two DataFrames containing the testing data with columns id, table_name, label, label_probs
    """
    logging.info("Getting train and test data ...")
    try:
        with open(train_path, 'rb') as infile:
            train_df = pd.read_pickle(infile).reset_index(drop=True)
        with open(test_path, 'rb') as infile:
            test_df = pd.read_pickle(infile).reset_index(drop=True)

    except IOError as e:
        sys.exit("Could not read data: {}".format(e.args))

    return train_df, test_df


def binarize_labels(y_train: [[str]], y_test: [[str]], y_gold: [[str]] = None) -> (MultiLabelBinarizer,
                                                                                   [[int]],
                                                                                   [[int]],
                                                                                   [[int]]):
    """
    This function takes the list of label names and turns it into a one-hot encoding for each data point, then returns
    the binarizer and transformed labels for training and testing
    :param y_train: list of labels such as [['cat'],['dog','bird']]
    :param y_test: list of labels from label model such as [['cat'],['dog','bird']]
    :param y_gold: list of labels from manual annotation such as [['cat'],['dog','bird']]

    """
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)
    if y_gold:
        y_gold = mlb.transform(y_gold)
    return mlb, y_train, y_test, y_gold


def ravel_inverse_binarized_labels(mlb: MultiLabelBinarizer, y: [[int]]) -> [str]:
    """
    Used to get a flattened array of labels for some multiclass evaluations.
    e.g.
    y = array([[0, 1, 0],
               [0, 0, 1],
               [1, 0, 0],
               [0, 1, 0]])
    np.ravel(mlb.inverse_transform(y)) -> array(['cat', 'dog', 'bird', 'cat'], dtype='<U4')
    """
    return np.ravel(mlb.inverse_transform(y))


def check_if_multiclass(y_train: [[int]], y_test: [[int]]) -> (bool, bool):
    """
    count the values in each row that are non-zero e.g. np.count_nonzero([[0,1,0],[0,0,1], axis=1) -> [1,1]
    then check to see if all elements in the result are equal to 1 e.g. [1,1] == 1 -> [True, True]
    then make sure the whole array is true e.g. np.all([True,True]) -> True

    i.e., there are no rows with more than one 1. If false, then the task is a multilabel classification
    """
    train_is_multiclass = np.all(np.count_nonzero(y_train, axis=1) == 1)
    test_is_multiclass = np.all(np.count_nonzero(y_test, axis=1) == 1)
    return train_is_multiclass, test_is_multiclass


def get_train_features(train_df: pd.DataFrame, features: [str]) -> np.ndarray:
    features_df = pd.DataFrame()
    for tbl in train_df.table_name.unique():
        tbl_df = train_df.loc[(train_df.table_name == tbl)]
        tbl_f_df = pd.read_sql(SQL_QUERY.format(column=', '.join(['id'] + features),
                                                table=tbl,
                                                ids=str(tuple(tbl_df.id.tolist()))),
                               DATABASE_IP,
                               chunksize=100)
        for data in tbl_f_df:
            features_df = features_df.append(data, ignore_index=True)
    id_f_df = pd.merge(train_df, features_df, on='id')
    logging.info("Training features loaded. Here's a peek:")
    logging.info(id_f_df.head())
    feature_arrays = [np.array(id_f_df[feature].tolist()) for feature in features]
    try:
        x = np.concatenate(feature_arrays, axis=1)
    except np.AxisError:
        x = feature_arrays[0]
    return x


def save_df(df: pd.DataFrame, pkl_filename: str, html_filename: str):
    logging.info("Here's a preview of {}:".format(pkl_filename))
    logging.info(df.head())
    df.to_pickle(pkl_filename)
    df.head().to_html(html_filename)


def save_training_features(x_train: np.ndarray, filename: str):
    np.save(filename, x_train)
