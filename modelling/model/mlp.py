"""
multi-layer perceptron classifier
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
"""
import argparse
import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import sklearn.metrics as eval
from sklearn.neural_network import MLPClassifier

from model import (X_TRAIN_FILENAME, TRAIN_DF_FILENAME, TRAIN_DF_HTML_FILENAME, TEST_PRED_DF_FILENAME,
                   TEST_PRED_DF_HTML_FILENAME, CONFUSION_MATRIX_FILENAME, LOGGING_FILENAME)
import model.data as data
import model.tracking as tracking
from model_objs import LookupClassifier

parser = argparse.ArgumentParser(description='Train a multi-layer perceptron classifier.')
parser.add_argument('train_path', type=str, help='File path or URL to the training data')
parser.add_argument('test_path', type=str, help='File path or URL to the test data')
parser.add_argument('features', nargs='+', type=str, help='column name(s) of the features to use.')

parser.add_argument('--activation', choices=('identity', 'logistic', 'tanh', 'relu'), default='relu', type=str,
                    help='Activation function for the hidden layer.')
parser.add_argument('--solver', choices=('adam', 'lbfgs', 'sgd'), default='adam', type=str,
                    help='The solver for weight optimization.')
parser.add_argument('--alpha', default=0.0001, type=float, help='L2 penalty (regularization term) parameter.')
parser.add_argument('--learning_rate', choices=('constant', 'invscaling', 'adaptive'), default='constant', type=str,
                    help='Learning rate schedule for weight updates.')
parser.add_argument('--learning_rate_init', default=0.001, type=float,
                    help='The initial learning rate used. It controls the step-size in updating the weights.')
parser.add_argument('--power_t', default=0.5, type=float, help='The exponent for inverse scaling rate.')
parser.add_argument('--max_iter', default=200, type=int, help='Maximum number of iterations.')
parser.add_argument('--tol', default=1e-4, type=float, help='Tolerance for the optimization.')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for gradient descent update.')
parser.add_argument('--beta_1', default=0.9, type=float,
                    help='Exponential decay rate for estimates of first moment vector in adam.')
parser.add_argument('--beta_2', default=0.999, type=float,
                    help='Exponential decay rate for estimates of second moment vector in adam.')
parser.add_argument('--epsilon', default=1e-8, type=float, help='Value for numerical stability in adam.')
parser.add_argument('--n_iter_no_change', default=10, type=int,
                    help='Maximum number of epochs to not meet tol improvement.')
parser.add_argument('--max_fun', default=15000, type=int, help='Maximum number of loss function calls.')

parsed_args = parser.parse_args()

TRAIN_PARAMS = {
    'activation': parsed_args.activation,
    'solver': parsed_args.solver,
    'alpha': parsed_args.alpha,
    'max_iter': parsed_args.max_iter,
    'tol': parsed_args.tol,
    'verbose': True
}
TRAIN_PARAMS.update(
    {
        'adam': {
            'learning_rate_init': parsed_args.learning_rate_init,
            'early_stopping': True,
            'beta_1': parsed_args.beta_1,  # keep this between 0 and 1
            'beta_2': parsed_args.beta_2,  # keep this between 0 and 1
            'epsilon': parsed_args.epsilon,
            'n_iter_no_change': parsed_args.n_iter_no_change
        },
        'lbfgs': {
            'max_fun': parsed_args.max_fun
        },
        'sgd': {
            'learning_rate': parsed_args.learning_rate,
            'learning_rate_init': parsed_args.learning_rate_init,
            'power_t': parsed_args.power_t,
            'momentum': parsed_args.momentum,  # keep this between 0 and 1
            'early_stopping': True,
            'n_iter_no_change': parsed_args.n_iter_no_change
        }
    }[TRAIN_PARAMS['solver']])

REGISTERED_MODEL_NAME = '{}_MLP'.format('_'.join(parsed_args.features))
ARTIFACT_PATH = 'mlp'


def evaluate_multiclass(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_prob: np.ndarray,
                        ravel_y_true: [str],
                        ravel_y_pred: [str],
                        x_test: np.ndarray,
                        mlp_model: LookupClassifier) -> dict:
    metrics_dict = common_metrics(y_true, y_pred, y_prob)
    metrics_dict.update(binary_metrics())
    metrics_dict.update({
        'F1 micro': float(eval.f1_score(y_true, y_pred, average='micro')),
        'F1 macro': float(eval.f1_score(y_true, y_pred, average='macro')),
        'F1 weighted': float(eval.f1_score(y_true, y_pred, average='weighted')),
        'balanced accuracy score': eval.balanced_accuracy_score(ravel_y_true, ravel_y_pred),

    })
    eval.plot_confusion_matrix(mlp_model.classifier, x_test, ravel_y_true, labels=mlp_model.classes)
    plt.savefig(CONFUSION_MATRIX_FILENAME)
    plt.close()
    return metrics_dict


def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics_dict = common_metrics(y_true, y_pred, y_prob)
    metrics_dict.update({
        'discounted cumulative gain': eval.dcg_score(y_true, y_prob)
    })


def common_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    functions that can be used for multiclass or multilabel.
    """
    metrics_dict = {'accuracy': eval.accuracy_score(y_true, y_pred),
                    'macro average precision score': eval.average_precision_score(y_true, y_prob)}
    return metrics_dict


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, ravel_y_true: [str], ravel_y_pred: [str]) -> dict:
    flat_max_probs = [max(probs) for probs in y_prob]
    metrics_dict = {

    }

def main():
    with mlflow.start_run():
        logging.info("Loading training and testing sets, and feature sets")
        train_df, test_df = data.load(parsed_args.train_path, parsed_args.test_path)
        data.save_df(train_df, TRAIN_DF_FILENAME, TRAIN_DF_HTML_FILENAME)

        x_train = data.get_train_features(train_df, parsed_args.features)
        data.save_training_features(x_train, X_TRAIN_FILENAME)

        logging.info("Encoding labels and determining multiclass or multilabel ...")
        mlb, y_train, y_test = data.binarize_labels(y_train=train_df.label.tolist(),
                                               y_test=test_df.label.tolist())
        train_is_multiclass, test_is_multiclass = data.check_if_multiclass(y_train, y_test)

        logging.info("Training mlp ...")
        mlp = MLPClassifier(**TRAIN_PARAMS)
        mlp.fit(x_train, y_train)
        mlp_model = LookupClassifier(mlb, mlp, parsed_args.features)

        logging.info("Predicting test ...")
        test_pred_df = mlp_model.predict(test_df)
        y_pred = test_pred_df[mlp_model.classes].to_numpy()
        y_prob = test_pred_df[mlp_model.prob_labels].to_numpy()

        logging.info("Evaluating model ...")
        try:
            if train_is_multiclass and test_is_multiclass:
                logging.info("Train and test are multiclass. Using multiclass evaluation ...")
                ravel_y_test = data.ravel_inverse_binarized_labels(mlb, y_test)
                ravel_y_pred = data.ravel_inverse_binarized_labels(mlb, y_pred)
                x_test = mlp_model.get_features(test_pred_df)
                metrics = evaluate_multiclass(y_test, y_pred, y_prob, ravel_y_test, ravel_y_pred)
            elif not train_is_multiclass and not test_is_multiclass:
                logging.info("Train and test are multilabel. Using multilabel evaluation ...")
                metrics = evaluate_multilabel(y_test, y_pred, y_prob)
            else:
                logging.warning("""Train and test do not have matching classification types. 
                This could have consequences in evaluation. Trying multilabel evaluation ...""")
                try:
                    metrics = evaluate_multilabel(y_test, y_pred, y_prob)
                except Exception as e:
                    msg = "Unable to perform multilabel evaluation:\n{}\nTrying multiclass ...".format(e.args)
                    logging.error(msg)
                    try:
                        metrics = evaluate_multiclass(y_test, y_pred, y_prob)
                    except Exception as e:
                        msg = "Unable to perform multiclass evaluation:\n{}\nStopping.".format(e.args)
                        logging.error(msg)
                        metrics = None
        except Exception as e:
            msg = "Unable to perform evaluation:\n{}".format(e.args)
            logging.error(msg)
            metrics = None

        logging.info("Logging artifacts and saving model ...")
        log_artifacts(parsed_args.train_path, parsed_args.test_path)


if __name__ == '__main__':
    main()
