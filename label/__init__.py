import argparse
import logging
import os
import sys
import pandas as pd

parser = argparse.ArgumentParser(description='Perform data programming.')
parser.add_argument('train_data', help='File path or URL to the unlabeled training data')
parser.add_argument('--task', default='multilabel', type=str, help='classification task (multiclass or multilabel)')
parser.add_argument('--dev_data', default=0, help='Use labeled development data for training and evaluation')
parser.add_argument('--n_epochs', default=1000, type=int, help='the number of epochs to train the Label Model (where '
                                                               'each epoch is a single optimization step)')
parser.add_argument('--optimizer', default='sgd', help='which optimizer to use for the Label Model')
parser.add_argument('--prec_init', default=0.7, type=float, help='LF precision initializations / priors')
parsed_args = parser.parse_args()

TMP_ARTIFACTS = '/tmp_artifacts'

LOGGING_FILENAME = os.path.join(TMP_ARTIFACTS, 'log.txt')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=LOGGING_FILENAME, filemode='w')

logging.info("Loading unlabeled training data ...")
TRAIN_DF_FILENAME = parsed_args.train_data
try:
    TRAIN_DF = pd.read_pickle(TRAIN_DF_FILENAME)
except IOError:
    sys.exit("Invalid path to data")

TRAIN_PARAMS = {
    'n_epochs': parsed_args.n_epochs,
    'optimizer': parsed_args.optimizer,
    'prec_init': parsed_args.prec_init
}

CRATE_DB_IP = os.environ['CRATE_DB_IP']

LABEL_MATRIX_FILENAME = os.path.join(TMP_ARTIFACTS, 'label_matrix.npy')
LABEL_MODEL_FILENAME = os.path.join(TMP_ARTIFACTS, 'label_model.pkl')

LF_SUMMARY_FILENAME = os.path.join(TMP_ARTIFACTS, 'lf_summary.html')
CONFUSION_MATRIX_FILENAME = os.path.join(TMP_ARTIFACTS, 'confusion_matrix.jpg')

TRAINING_DATA_FILENAME = os.path.join(TMP_ARTIFACTS, 'training_data.pkl')
TRAINING_DATA_HTML_FILENAME = os.path.join(TMP_ARTIFACTS, 'training_data.html')
