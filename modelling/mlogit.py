"""
References:
    https://keras.io/examples/imdb_bidirectional_lstm/
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highligh:logistic%20regression#sklearn.linear_model.LogisticRegression
"""
from modelling import (X_TRAIN_FILENAME, TRAIN_DF_HTML_FILENAME, TEST_DF_FILENAME, TEST_DF_HTML_FILENAME,
                       ROC_CURVE_FILENAME, CONFUSION_MATRIX_FILENAME)
from modelling.bundle import MlogitModelBundle
import pickle
import pandas as pd
import argparse
import sys
import logging
from urllib.parse import urlparse

import mlflow.tensorflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, f1_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train a classifier.')
parser.add_argument('train_path', help='File path or URL to the training data')
parser.add_argument('test_path', help='File path or URL to the test data')

parser.add_argument('--solver', default='lbfgs', help='Algorithm to use in the optimization problem.')
parser.add_argument('--tol', default=1e-4, type=float, help='Tolerance for stopping criteria')
parser.add_argument('--C', default=1.0, type=float, help='Inverse of regularization strength; must be a positive '
                                                         'float. Like in support vector machines, smaller values '
                                                         'specify stronger regularization')
parser.add_argument('--max_iter', default=100, type=int, help='Maximum number of iterations taken for the solvers to '
                                                              'converge')
parser.add_argument('--multi_class', default='auto', help='If the option chosen is ‘ovr’, then a binary problem '
                                                          'is fit for each label. For ‘multinomial’ the loss '
                                                          'minimised is the multinomial loss fit across the '
                                                          'entire probability distribution, even when the data is '
                                                          'binary')
parser.add_argument('--verbose', default=10000, type=int, help='For the liblinear and lbfgs solvers set verbose to any '
                                                               'positive number for verbosity.')
parser.add_argument('--n_jobs', default=-1, type=int,
                    help='Number of CPU cores used when parallelizing over classes if '
                         'multi_class=’ovr’.This parameter is ignored when the solver is '
                         'set to ‘liblinear’ regardless of whether ‘multi_class’ is '
                         'specified or not. None means 1 unless in a '
                         'joblib.parallel_backend context. -1 means using all processors')
parsed_args = parser.parse_args()

"""
Penalty: Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 
 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no 
 regularization is applied.

Dual: Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. 
 Prefer dual=False when n_samples > n_features.

Intercept Scaling: Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case,
 x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is 
 appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight. 
 Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of 
 regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.

Random State: Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data.

L1 Ratio: The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. 
 Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. 
 For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.
"""
TRAIN_PARAMS = {
    'solver': parsed_args.solver,
    'tol': parsed_args.tol,
    'C': parsed_args.C,
    'max_iter': parsed_args.max_iter,
    'multi_class': parsed_args.multi_class,
    'verbose': parsed_args.verbose,
    'n_jobs': parsed_args.n_jobs
}
TRAIN_PARAMS.update(
    {
        'newton-cg': {
            'penalty': 'l2'
        },
        'lbfgs': {
            'penalty': 'l2',
            'verbose': 10000
        },
        'liblinear': {
            'penalty': 'l1',
            'fit_intercept': True,
            'intercept_scaling': 1,
            'random_state': 42,
            'multi_class': 'ovr',
            'verbose': 10000
        },
        'sag': {
            'penalty': 'l2',
            'random_state': 42,
        },
        'saga': {
            'penalty': 'elasticnet',
            'random_state': 42,
            'l1_ratio': 0.7
        }
    }[TRAIN_PARAMS['solver']])

REGISTERED_MODEL_NAME = 'MLogitBundle'


def evaluate_model(bundle, x_test, y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    plot_roc_curve(bundle.model, x_test, y_true)
    plt.savefig(ROC_CURVE_FILENAME)
    plt.close()
    plot_confusion_matrix(bundle.model, x_test, y_true)
    plt.savefig(CONFUSION_MATRIX_FILENAME)
    plt.close()
    return {'F1': f1}


def main():
    with mlflow.start_run():
        logging.info("Getting train and test data ...")
        try:
            with open(parsed_args.train_path, 'rb') as infile:
                train_df = pd.read_pickle(infile)
            with open(parsed_args.test_path, 'rb') as infile:
                test_df = pd.read_pickle(infile)
        except IOError:
            sys.exit("Could not read data")

        y_train = train_df.label.tolist()
        y_test = test_df.label.tolist()

        mlogit_bundle = MlogitModelBundle(TRAIN_PARAMS)

        logging.info("Training mlogit bundle ...")
        x_train = mlogit_bundle.train(train_df.text.tolist(), y_train)
        x_test = mlogit_bundle.featurize(mlogit_bundle.preprocess(test_df.text.tolist()))

        logging.info("Evaluating model ...")
        y_pred = mlogit_bundle.predict(test_df.text.tolist())
        metrics = evaluate_model(mlogit_bundle, x_test, y_test, y_pred)
        mlflow.log_metrics(metrics)

        logging.info("Saving bundle ...")
        signature = infer_signature(x_test, y_pred)
        input_example = x_train[:5, :]
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != 'file':
            # mlogit bundle subclasses tf.Module
            mlflow.tensorflow.log_model(
                mlogit_bundle,
                'mlogit_bundle',
                registered_model_name=REGISTERED_MODEL_NAME,
                signature=signature,
                input_example=input_example)
        else:
            mlflow.tensorflow.log_model(mlogit_bundle, 'mlogit_bundle')

        logging.info("Saving artifacts ...")
        test_df.insert(len(test_df.columns), 'pred', y_pred)
        pd.to_pickle(test_df, TEST_DF_FILENAME)
        test_df.to_html(TEST_DF_HTML_FILENAME)

        mlflow.log_params(mlogit_bundle.model.get_params(deep=False))
        mlflow.log_artifact(parsed_args.train_path)
        mlflow.log_artifact(parsed_args.test_path)
        mlflow.log_artifact(TEST_DF_FILENAME)
        mlflow.log_artifact(TEST_DF_HTML_FILENAME)
        mlflow.log_artifact(ROC_CURVE_FILENAME)
        mlflow.log_artifact(CONFUSION_MATRIX_FILENAME)


if __name__ == '__main__':
    main()
