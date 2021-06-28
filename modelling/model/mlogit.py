"""
References:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highligh:logistic%20regression#sklearn.linear_model.LogisticRegression
"""
import argparse
import logging

import matplotlib.pyplot as plt
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, f1_score

from model import CONFUSION_MATRIX_FILENAME
from model.data import load, get_train_features, binarize_labels, predict_test, log_artifacts
from model_objs import LookupClassifier, save_model

parser = argparse.ArgumentParser(description='Train a multinomial logistic regression classifier.')
parser.add_argument('train_path', type=str, help='File path or URL to the training data')
parser.add_argument('test_path', type=str, help='File path or URL to the test data')
parser.add_argument('features', default='tweet_use', nargs='+', type=str, help='column name(s) of the features to use.')

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

REGISTERED_MODEL_NAME = '{}_MLogit'.format('_'.join(parsed_args.features))
ARTIFACT_PATH = 'mlogit'

# mlflow.sklearn.autolog()


def evaluate_model(pipe, x_test, y_true, y_pred):
    f1 = {'F1 micro': float(f1_score(y_true, y_pred, average='micro')),
          'F1 macro': float(f1_score(y_true, y_pred, average='macro')),
          'F1 weighted': float(f1_score(y_true, y_pred, average='weighted'))}
    # plot_roc_curve(pipe, x_test, y_true)
    # plt.savefig(ROC_CURVE_FILENAME)
    # plt.close()
    plot_confusion_matrix(pipe, x_test, y_true)
    plt.savefig(CONFUSION_MATRIX_FILENAME)
    plt.close()
    return f1


def main():
    with mlflow.start_run():
        logging.info("Loading training and testing sets, and feature sets")
        train_df, test_df = load(parsed_args.train_path, parsed_args.test_path)
        x_train = get_train_features(train_df, parsed_args.features)
        mlb, y_train, y_test = binarize_labels(y_train=train_df.label.tolist(),
                                               y_test=test_df.label.tolist(),
                                               return_inverse=True)

        logging.info("Training mlogit ...")
        mlogit = LogisticRegression(**TRAIN_PARAMS)
        mlogit.fit(x_train, y_train)
        mlogit_model = LookupClassifier(mlb, mlogit, parsed_args.features, used_inverse_labels=True)

        logging.info("Predicting test ...")
        test_pred_df = predict_test(mlogit_model, test_df)

        logging.info("Saving model ...")
        save_model(x_train, test_pred_df, mlogit_model, ARTIFACT_PATH, REGISTERED_MODEL_NAME)

        # logging.info("Evaluating model ...")

        # metrics = evaluate_multiclass(mlp_model, x_test, y_test, [list(row) for row in y_pred_df.iterrows()])
        # mlflow.log_metrics(metrics)

        logging.info("Logging artifacts ...")
        log_artifacts(parsed_args.train_path, parsed_args.test_path)


if __name__ == '__main__':
    main()
