import os
import logging

TMP_ARTIFACTS = '/tmp_artifacts'

X_TRAIN_FILENAME = os.path.join(TMP_ARTIFACTS, 'x_train.pkl')
BUNDLED_FILENAME = os.path.join(TMP_ARTIFACTS, 'bundled.pkl')
TRAIN_DF_HTML_FILENAME = os.path.join(TMP_ARTIFACTS, 'train.html')
TEST_DF_FILENAME = os.path.join(TMP_ARTIFACTS, 'test.pkl')
TEST_DF_HTML_FILENAME = os.path.join(TMP_ARTIFACTS, 'test.html')
ROC_CURVE_FILENAME = os.path.join(TMP_ARTIFACTS, 'roc.jpg')
CONFUSION_MATRIX_FILENAME = os.path.join(TMP_ARTIFACTS, 'confusion_matrix.jpg')
MLB_FILENAME = os.path.join(TMP_ARTIFACTS, 'mlb.pkl')

LOGGING_FILENAME = os.path.join(TMP_ARTIFACTS, 'log.txt')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=LOGGING_FILENAME, filemode='w')
