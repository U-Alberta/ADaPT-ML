import mlflow
import logging
from label import (LOGGING_FILENAME, TRAIN_DF_FILENAME, LABEL_MATRIX_FILENAME, TRAINING_DATA_FILENAME,
                   TRAINING_DATA_HTML_FILENAME, LF_SUMMARY_FILENAME, CONFUSION_MATRIX_FILENAME)
import mlflow.pytorch


def log(train_params, metrics, signature, input_example, model_name, label_model):

    # Log the data points, label matrix, and labeled training data as artifacts
    mlflow.log_params(train_params)
    try:
        mlflow.log_metrics(metrics)
    except:
        logging.warning("metrics not available.")

    # LabelModel subclasses torch.nn.Module
    mlflow.pytorch.log_model(
        label_model,
        'label_model',
        registered_model_name=model_name,
        signature=signature,
        input_example=input_example)

    try:
        mlflow.log_artifact(LOGGING_FILENAME)
        mlflow.log_artifact(TRAIN_DF_FILENAME)
        mlflow.log_artifact(LABEL_MATRIX_FILENAME)
        mlflow.log_artifact(TRAINING_DATA_FILENAME)
        mlflow.log_artifact(TRAINING_DATA_HTML_FILENAME)
        mlflow.log_artifact(LF_SUMMARY_FILENAME)
        mlflow.log_artifact(CONFUSION_MATRIX_FILENAME)
    except:
        logging.warning("an artifact could not be logged")
