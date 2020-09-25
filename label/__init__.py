import argparse
import logging
import os

parser = argparse.ArgumentParser(description='Perform data programming.')
parser.add_argument('data_path', help='File path or URL to the unlabeled training data')
parser.add_argument('--step', default=0, type=int, help='create label matrix (1), label model (2), training data (3). '
                                                        'or all (0)')
parser.add_argument('--eval', default=0, help='evaluate labeling functions (1), label model (2), or both (0)')
parser.add_argument('--n_epochs', default=1000, type=int, help='the number of epochs to train the Label Model (where '
                                                               'each epoch is a single optimization step)')
parser.add_argument('--optimizer', default='sgd', help='which optimizer to use for the Label Model')
parser.add_argument('--prec_init', default=0.7, type=float, help='LF precision initializations / priors')
parsed_args = parser.parse_args()

TRAIN_PARAMS = {
    'n_epochs': parsed_args.n_epochs,
    'optimizer': parsed_args.optimizer,
    'prec_init': parsed_args.prec_init
}

LABEL_MATRIX_FILENAME = os.path.join('label', 'resources', 'label_matrix.npy')
LABEL_MODEL_FILENAME = os.path.join('label', 'resources', 'label_model.pkl')
REGISTERED_MODEL_NAME = 'ValueLabelModel'

LF_SUMMARY_FILENAME = 'lf_summary.html'
CONFUSION_MATRIX_FILENAME = 'confusion_matrix.jpg'

TRAINING_DATA_FILENAME = os.path.join('label', 'data', 'training_data.pkl')
TRAINING_DATA_HTML_FILENAME = os.path.join('label', 'data', 'training_data.html')

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
