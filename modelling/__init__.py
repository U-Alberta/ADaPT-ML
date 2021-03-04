import os
import logging


TMP_ARTIFACTS = '/tmp_artifacts'
CRATE_DB_IP = os.environ['CRATE_DB_IP']
SQL_QUERY = """
    SELECT {column} FROM {table} WHERE id IN {ids};
    """

X_TRAIN_FILENAME = os.path.join(TMP_ARTIFACTS, 'x_train.npy')
TRAIN_DF_HTML_FILENAME = os.path.join(TMP_ARTIFACTS, 'train.html')
TEST_PRED_DF_FILENAME = os.path.join(TMP_ARTIFACTS, 'test.pkl')
TEST_PRED_DF_HTML_FILENAME = os.path.join(TMP_ARTIFACTS, 'test.html')
ROC_CURVE_FILENAME = os.path.join(TMP_ARTIFACTS, 'roc.jpg')
CONFUSION_MATRIX_FILENAME = os.path.join(TMP_ARTIFACTS, 'confusion_matrix.jpg')

LOGGING_FILENAME = os.path.join(TMP_ARTIFACTS, 'log.txt')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=LOGGING_FILENAME, filemode='w')
