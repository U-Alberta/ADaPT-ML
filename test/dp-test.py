import subprocess
import pandas as pd
import os
import sys

MLFLOW_BEGIN = ["wait-for-it", "dp-mlflow-db:3306", "-s", "--",
                "mlflow", "run", "--no-conda", "-e", "example", "--experiment-name", "test"]
MLFLOW_END = ["-P", "seed=8", "."]
# variables for artifacts
DEV_ARTIFACTS = ["dev_label_matrix.npy", "development_data.html", "development_data.pkl", "lf_summary_dev.html"]
BASE_ARTIFACTS = ["label_model.pkl", "lf_summary_train.html", "log.txt", "train_label_matrix.npy", "training_data.html",
                  "training_data.pkl"]
MULTILABEL_DEV_ARTIFACTS = BASE_ARTIFACTS + DEV_ARTIFACTS
MULTICLASS_DEV_ARTIFACTS = BASE_ARTIFACTS + DEV_ARTIFACTS + ["confusion_matrix.jpg"]
# index in df for each run
MULTICLASS_DEV_I = 2
MULTILABEL_DEV_I = 0


def test_run(params):
    try:
        with open('stdout_log.txt', 'w') as stdout_log:
            subprocess.run(MLFLOW_BEGIN + params + MLFLOW_END, check=True, stdout=stdout_log, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("ERROR: Experiment run failed to complete.")
        sys.exit(1)


print("=== STARTING DATA PROGRAMMING TESTS WITH EXAMPLE DATA ===")

### Try running all experiments and see if they are successful ###

print("Testing multiclass...")
test_run(["-P", "train_data=/unlabeled_data/multiclass_df.pkl",
          "-P", "dev_data=0",
          "-P", "task=multiclass"])

print("Testing multiclass with development data...")
os.rename("/annotations/example/multiclass_gold_df.pkl", "/annotations/example/gold_df.pkl")
try:
    test_run(["-P", "train_data=/unlabeled_data/multiclass_df.pkl",
              "-P", "dev_data=1",
              "-P", "task=multiclass"])
finally:
    os.rename("/annotations/example/gold_df.pkl", "/annotations/example/multiclass_gold_df.pkl")

print("Testing multilabel...")
test_run(["-P", "train_data=/unlabeled_data/multilabel_df.pkl",
          "-P", "dev_data=0",
          "-P", "task=multilabel"])

print("Testing multilabel with development data...")
os.rename("/annotations/example/multilabel_gold_df.pkl", "/annotations/example/gold_df.pkl")
try:
    test_run(["-P" "train_data=/unlabeled_data/multilabel_df.pkl",
              "-P", "dev_data=1",
              "-P", "task=multilabel"])
finally:
    os.rename("/annotations/example/gold_df.pkl", "/annotations/example/multilabel_gold_df.pkl")

### All experiments were successful, so check the artifacts and logs ###

subprocess.run(["mlflow", "experiments", "csv", "-x", "1", "-o", "./runs.csv"], check=True)
runs_df = pd.read_csv("./runs.csv").sort_values('start_time', ascending=False).reset_index(drop=True).head(4)
os.remove("./runs.csv")

for run in runs_df.itertuples():

    print("Checking results for run {}".format(run.run_id))
    try:
        for artifact in BASE_ARTIFACTS:
            assert os.path.exists(os.path.join(run.artifact_uri, artifact))
        if run.Index == MULTICLASS_DEV_I:
            for artifact in MULTICLASS_DEV_ARTIFACTS:
                assert os.path.exists(os.path.join(run.artifact_uri, artifact))
        elif run.Index == MULTILABEL_DEV_I:
            for artifact in MULTILABEL_DEV_ARTIFACTS:
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

print("=== DATA PROGRAMMING TESTS WITH EXAMPLE DATA COMPLETED SUCCESSFULLY ===")
