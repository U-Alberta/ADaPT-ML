from label import (parsed_args, matrix, model,
                   TRAIN_CSV_FILENAME, TRAIN_DF_FILENAME, TRAIN_MATRIX_FILENAME,
                   LABEL_MODEL_FILENAME, TRAINING_DATA_FILENAME)
import mlflow
import mlflow.tensorflow as mltf
from urllib.parse import urlparse

if __name__ == '__main__':

    train_df = matrix.create_df(TRAIN_CSV_FILENAME, TRAIN_DF_FILENAME)

    # Step 1: create the label matrix
    if parsed_args.step in (0, 1):
        L_train = matrix.create_label_matrix(train_df, TRAIN_MATRIX_FILENAME)
    else:
        L_train = matrix.load_label_matrix(TRAIN_MATRIX_FILENAME)

    # Step 2: train the label model
    if parsed_args.step in (0, 2):
        with mlflow.start_run():
            label_model = model.train_label_model(L_train, LABEL_MODEL_FILENAME)

            mlflow.log_params(model.train_params_dict(label_model))
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != 'file':
                mltf.log_model(label_model, 'model', registered_model_name='ATSCLabelModel')
            else:
                mltf.log_model(label_model, 'model')
            mltf.save_model()
    else:
        label_model = model.load_label_model(LABEL_MODEL_FILENAME)

    # Step 3: apply the label model to the training data
    if parsed_args.step in (0, 3):
        filtered_train_df = model.apply_label_model(L_train, label_model, train_df, TRAINING_DATA_FILENAME)
