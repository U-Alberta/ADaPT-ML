import logging
import sys
from urllib.parse import urlparse

import mlflow.pytorch
import pandas as pd
from mlflow.models.signature import infer_signature

from label import (parsed_args, matrix, model, evaluate, LOGGING_FILENAME, LABEL_MATRIX_FILENAME,
                   TRAINING_DATA_FILENAME, TRAINING_DATA_HTML_FILENAME, LF_SUMMARY_FILENAME, REGISTERED_MODEL_NAME,
                   CONFUSION_MATRIX_FILENAME)

if __name__ == '__main__':

    with mlflow.start_run():
        logging.info("Loading unlabeled training data ...")
        try:
            train_df = pd.read_pickle(parsed_args.data_path)
        except IOError:
            sys.exit("Invalid path to training data: {0}".format(parsed_args.data_path))

        # Step 1: create the label matrix
        if parsed_args.step in (0, 1):
            L_train = matrix.create_label_matrix(train_df)
        else:
            L_train = matrix.load_label_matrix()

        # Step 2: train the label model
        if parsed_args.step in (0, 2):
            label_model = model.train_label_model(L_train)
            mlflow.log_params(model.train_params_dict(label_model))
            #metrics = evaluate.label_model_summary(L_train, label_model, y)
            #mlflow.log_metrics(metrics)
            evaluate.lf_summary(L_train, label_model)

            signature = infer_signature(L_train, label_model.predict(L_train))
            input_example = L_train[:5, :]
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != 'file':
                # LabelModel subclasses torch.nn.Module
                mlflow.pytorch.log_model(
                    label_model,
                    'label_model',
                    registered_model_name=REGISTERED_MODEL_NAME,
                    signature=signature,
                    input_example=input_example)
            else:
                mlflow.pytorch.log_model(label_model, 'label_model')
        else:
            label_model = model.load_label_model()

        # Step 3: apply the label model to the training data
        if parsed_args.step in (0, 3):
            filtered_train_df = model.apply_label_model(L_train, label_model, train_df)

        # Log the data points, label matrix, and labeled training data as artifacts
        mlflow.log_artifact(LOGGING_FILENAME)
        mlflow.log_artifact(parsed_args.data_path)
        mlflow.log_artifact(LABEL_MATRIX_FILENAME)
        mlflow.log_artifact(TRAINING_DATA_FILENAME)
        mlflow.log_artifact(TRAINING_DATA_HTML_FILENAME)
        mlflow.log_artifact(LF_SUMMARY_FILENAME)
        # mlflow.log_artifact(CONFUSION_MATRIX_FILENAME)
