import argparse
import os
import logging

X_TRAIN_FILENAME = os.path.join('modeling', 'resources', 'x_train.pkl')
BUNDLED_FILENAME = os.path.join('classify', 'resources', 'bundled.pkl')
TRAIN_DF_HTML_FILENAME = os.path.join('classify', 'resources', 'train.html')
TEST_DF_FILENAME = os.path.join('classify', 'resources', 'test.pkl')
TEST_DF_HTML_FILENAME = os.path.join('classify', 'resources', 'test.html')
ROC_CURVE_FILENAME = os.path.join('classify', 'resources', 'roc.jpg')
CONFUSION_MATRIX_FILENAME = os.path.join('classify', 'resources', 'confusion_matrix.jpg')

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
