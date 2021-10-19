import os
import pprint as pp
from model import TMP_ARTIFACTS
from glob import glob
import mlflow
import logging


def log(train_params, metrics):
    filenames = glob(os.path.join(TMP_ARTIFACTS, '*'))

    # Log the data points, label matrix, and labeled training data as artifacts
    # mlflow.log_params(train_params)
    try:
        mlflow.log_metrics(metrics)
        logging.info("Metrics:\n{}".format(pp.pformat(metrics)))
    except Exception as e:
        logging.warning("Metrics not logged:\n{}\n".format(e.args))

    # log all of the artifacts kept in the temporary folder into the proper run folder and remove them once logged
    for filename in filenames:
        try:
            mlflow.log_artifact(filename)
            os.remove(filename)
        except:
            print("An artifact could not be logged: {}".format(filename))


def save_model(x_train, model, registered_model_name, artifact_path):
    input_example = x_train[:5]
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=model,
        conda_env=model.conda_env,
        registered_model_name=registered_model_name,
        input_example=input_example
    )