import subprocess
import sys
import pandas as pd
import os

MLFLOW_BEGIN = ["wait-for-it", "modelling-mlflow-db:3306", "-s", "--",
                "mlflow", "run", "--no-conda", "-e", "mlp", "--experiment-name", "test"]
MLFLOW_END = ["-P", "features=txt_use", "-P", "solver=lbfgs", "-P", "random_state=8", "."]
# variables for artifacts
ARTIFACTS = ["confusion_matrix.jpg", "test.html", "test.pkl", "train.pkl", "train.html", "log.txt", "x_train.npy"]


def test_run(params):
    try:
        with open('stdout_log.txt', 'w') as stdout_log:
            subprocess.run(MLFLOW_BEGIN + params + MLFLOW_END, check=True, stdout=stdout_log, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("ERROR: Experiment run failed to complete.")
        sys.exit(1)


print("=== STARTING MODELLING TESTS WITH EXAMPLE DATA ===")

print("Testing multiclass with LM test data...")
test_run(["-P", "train_data=/dp_mlruns/multiclass_training_data.pkl",
          "-P", "test_data=/dp_mlruns/multiclass_training_data.pkl"])

print("Testing multiclass with gold test data...")
test_run(["-P", "train_data=/dp_mlruns/multiclass_training_data.pkl",
          "-P", "test_data=/dp_mlruns/multiclass_development_data.pkl"])

print("Testing multilabel with LM test data...")
test_run(["-P", "train_data=/dp_mlruns/multilabel_training_data.pkl",
          "-P", "test_data=/dp_mlruns/multilabel_training_data.pkl"])

print("Testing multilabel with gold test data...")
test_run(["-P", "train_data=/dp_mlruns/multilabel_training_data.pkl",
          "-P", "test_data=/dp_mlruns/multilabel_development_data.pkl"])

### All experiments were successful, so check the artifacts and logs ###

subprocess.run(["mlflow", "experiments", "csv", "-x", "1", "-o", "./runs.csv"], check=True)
runs_df = pd.read_csv("./runs.csv").sort_values('start_time', ascending=False).reset_index(drop=True).head(4)
os.remove("./runs.csv")

for run in runs_df.itertuples():

    print("Checking results for run {}".format(run.run_id))
    try:
        for artifact in ARTIFACTS:
            assert os.path.exists(os.path.join(run.artifact_uri, artifact))
    except AssertionError:
        print("ERROR: Artifact was not logged successfully for run {}.".format(run.run_id))
        sys.exit(1)

    try:
        subprocess.run(["grep", "-ni",
                        "-e", "\"WARNING\"",
                        "-e", "\"ERROR\"",
                        os.path.join(run.artifact_uri, 'log.txt')], check=True)
        print("ERROR: Problems with program execution found in the log file.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        pass

print("=== MODELLING TESTS WITH EXAMPLE DATA COMPLETED SUCCESSFULLY ===")
