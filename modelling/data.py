import pandas as pd
import numpy as np
import sys
import logging
import mlflow
from sklearn.preprocessing import MultiLabelBinarizer
from modelling import (X_TRAIN_FILENAME, TRAIN_DF_HTML_FILENAME, TEST_PRED_DF_FILENAME,
                       TEST_PRED_DF_HTML_FILENAME, CONFUSION_MATRIX_FILENAME, LOGGING_FILENAME)
from modelling import SQL_QUERY, CRATE_DB_IP, X_TRAIN_FILENAME


def load(train_path, test_path):
    logging.info("Getting train and test data ...")
    try:
        with open(train_path, 'rb') as infile:
            train_df = pd.read_pickle(infile)
            train_df.head().to_html(TRAIN_DF_HTML_FILENAME)
        with open(test_path, 'rb') as infile:
            test_df = pd.read_pickle(infile)
    except IOError:
        sys.exit("Could not read data")

    return train_df, test_df


def binarize_labels(y_train, y_test, return_inverse=False):
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)
    if return_inverse:
        y_train = np.ravel(mlb.inverse_transform(y_train))
        y_test = np.ravel(mlb.inverse_transform(y_test))
    return mlb, y_train, y_test


def get_train_features(train_df, features):
    train_features_df = pd.read_sql(SQL_QUERY.format(column=', '.join(features),
                                                     table=train_df.at[0, 'table'],
                                                     ids=str(tuple(train_df.id.tolist()))), CRATE_DB_IP)
    feature_arrays = [np.array(train_features_df[feature].tolist()) for feature in train_features_df]
    try:
        x_train = np.concatenate(feature_arrays, axis=1)
    except np.AxisError:
        x_train = feature_arrays[0]
    np.save(X_TRAIN_FILENAME, x_train)
    return x_train


def log_artifacts(train_path, test_path):
    mlflow.log_artifact(train_path)
    mlflow.log_artifact(test_path)
    mlflow.log_artifact(X_TRAIN_FILENAME)
    mlflow.log_artifact(TRAIN_DF_HTML_FILENAME)
    mlflow.log_artifact(TEST_PRED_DF_FILENAME)
    mlflow.log_artifact(TEST_PRED_DF_HTML_FILENAME)
    # mlflow.log_artifact(CONFUSION_MATRIX_FILENAME)
    mlflow.log_artifact(LOGGING_FILENAME)
