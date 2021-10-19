import os
import logging
import argparse


TMP_ARTIFACTS = '/tmp_artifacts'

X_TRAIN_FILENAME = os.path.join(TMP_ARTIFACTS, 'x_train.npy')

TRAIN_DF_FILENAME = os.path.join(TMP_ARTIFACTS, 'train.pkl')
TRAIN_DF_HTML_FILENAME = os.path.join(TMP_ARTIFACTS, 'train.html')

TEST_PRED_DF_FILENAME = os.path.join(TMP_ARTIFACTS, 'test.pkl')
TEST_PRED_DF_HTML_FILENAME = os.path.join(TMP_ARTIFACTS, 'test.html')

CONFUSION_MATRIX_FILENAME = os.path.join(TMP_ARTIFACTS, 'confusion_matrix.jpg')

LOGGING_FILENAME = os.path.join(TMP_ARTIFACTS, 'log.txt')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=LOGGING_FILENAME, filemode='w')

parser = argparse.ArgumentParser(description='Train a multi-layer perceptron classifier.')
parser.add_argument('train_path', type=str, help='File path or URL to the training data')
parser.add_argument('test_path', type=str, help='File path or URL to the test data')
parser.add_argument('features', nargs='+', type=str, help='column name(s) of the features to use.')
