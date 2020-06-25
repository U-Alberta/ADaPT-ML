import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Perform data programming.')
parser.add_argument('--step', default=0, help='create label matrix (1), label model (2), training data (3). or all (0)')
parser.add_argument('--eval', default=0, help='evaluate labeling functions (1), label model (2), or both (0)')
parsed_args = parser.parse_args()

TRAIN_CSV_FILENAME = os.path.join('label', 'data', 'unlabeled_train.csv')
