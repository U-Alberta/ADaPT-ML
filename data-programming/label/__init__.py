"""
Initialize the argument parser that will parse the arguments specified in the MLFlow command.
Initialize file paths for all of the artifacts that will be saved before they are logged in the appropriate MLFlow
    experiment folder
Load the training data that is specified in the arguments
"""


import argparse
import logging
import os

parser = argparse.ArgumentParser(description='Perform data programming.')
parser.add_argument('train_data', help='File path or URL to the unlabeled training data')
parser.add_argument('--task', default='multiclass', type=str, choices=('multiclass', 'multilabel'),
                    help='classification setting (multiclass or multilabel)')
parser.add_argument('--dev_data', default=0, type=int, choices=(0, 1),
                    help='Use labeled development data for training and evaluation?')
parser.add_argument('--n_epochs', default=1000, type=int,
                    help='the number of epochs to train the Label Model '
                    '(where each epoch is a single optimization step)')
parser.add_argument('--optimizer', default='sgd', choices=('sgd', 'adam', 'adamax'),
                    help='which optimizer to use for the Label Model')
parser.add_argument('--prec_init', default=0.7, type=float, help='LF precision initializations / priors')
parser.add_argument('--seed', default=0, type=int, help='a random seed to initialize the random number generator with')
parser.add_argument('--parallel', default=0, type=int, choices=(0, 1), help='run LFs in parallel?')
parser.add_argument('--device', default='cpu', type=str, choices=('cpu', 'cuda'),
                    help='config device to use for training the Label Model'),
parser.add_argument('--verbose', default=1, type=int, choices=(0, 1), help='redirect stdout to file?')

TMP_ARTIFACTS = '/tmp_artifacts'

TRAIN_MATRIX_FILENAME = os.path.join(TMP_ARTIFACTS, 'train_label_matrix.npy')
TRAINING_DATA_FILENAME = os.path.join(TMP_ARTIFACTS, 'training_data.pkl')
TRAINING_DATA_HTML_FILENAME = os.path.join(TMP_ARTIFACTS, 'training_data.html')

DEV_DF_FILENAME = os.path.join(TMP_ARTIFACTS, 'development_data.pkl')
DEV_DF_HTML_FILENAME = os.path.join(TMP_ARTIFACTS, 'development_data.html')
DEV_MATRIX_FILENAME = os.path.join(TMP_ARTIFACTS, 'dev_label_matrix.npy')

LABEL_MODEL_FILENAME = os.path.join(TMP_ARTIFACTS, 'label_model.pkl')

LF_SUMMARY_DEV_FILENAME = os.path.join(TMP_ARTIFACTS, 'lf_summary_dev.html')
LF_SUMMARY_TRAIN_FILENAME = os.path.join(TMP_ARTIFACTS, 'lf_summary_train.html')
CONFUSION_MATRIX_FILENAME = os.path.join(TMP_ARTIFACTS, 'confusion_matrix.jpg')

STDOUT_LOG_FILENAME = os.path.join(TMP_ARTIFACTS, 'stdout_log.txt')
LOGGING_FILENAME = os.path.join(TMP_ARTIFACTS, 'log.txt')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=LOGGING_FILENAME, filemode='w')

DATABASE_IP = os.environ['DATABASE_IP']
SQL_QUERY = """
    SELECT {column} FROM {table} WHERE id IN {ids};
    """
