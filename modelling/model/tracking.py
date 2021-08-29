import mlflow


def log(train_path, test_path):
    mlflow.log_artifact(train_path)
    mlflow.log_artifact(test_path)
    mlflow.log_artifact(X_TRAIN_FILENAME)
    mlflow.log_artifact(TRAIN_DF_HTML_FILENAME)
    mlflow.log_artifact(TEST_PRED_DF_FILENAME)
    mlflow.log_artifact(TEST_PRED_DF_HTML_FILENAME)
    # mlflow.log_artifact(CONFUSION_MATRIX_FILENAME)
    mlflow.log_artifact(LOGGING_FILENAME)


def save_model(x_train, test_pred_df, model, artifact_path, registered_model_name):
    # signature = infer_signature(x_train, test_pred_df)
    input_example = x_train[:5]
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=model,
        conda_env=model.conda_env,
        registered_model_name=registered_model_name,
        # signature=signature,
        input_example=input_example
    )