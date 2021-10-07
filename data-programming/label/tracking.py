"""
This module performs all MLFlow tracking tasks.
"""
import logging
import os
from glob import glob
import pprint as pp

import mlflow
import mlflow.pytorch

from label import TMP_ARTIFACTS


def log(metrics, input_example, model_name, label_model):
    filenames = glob(os.path.join(TMP_ARTIFACTS, '*'))

    # Log the data points, label matrix, and labeled training data as artifacts
    mlflow.log_params(train_params_dict(label_model))
    try:
        mlflow.log_metrics(metrics)
        logging.info("Metrics:\n{}".format(pp.pformat(metrics)))
    except Exception as err:
        if metrics is not None:
            logging.warning("Metrics failed to log:\n{}\n".format(err.args))

    # LabelModel subclasses torch.nn.Module
    mlflow.pytorch.log_model(
        label_model,
        'label_model',
        registered_model_name=model_name,
        input_example=input_example)

    # log all of the artifacts kept in the temporary folder into the proper run folder and remove them once logged
    for filename in filenames:
        try:
            mlflow.log_artifact(filename)
            os.remove(filename)
        except:
            print("An artifact could not be logged: {}".format(filename))


def train_params_dict(label_model) -> dict:
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
        logging.error("Label Model hasn't been trained yet...?")
