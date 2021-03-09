"""
multi-layer perceptron classifier
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
"""
import argparse
import logging

import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import plot_confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier

from modelling import CONFUSION_MATRIX_FILENAME
from modelling.data import load, get_train_features, binarize_labels, predict_test, log_artifacts
from model import LookupClassifier, save_model

parser = argparse.ArgumentParser(description='Train a multi-layer perceptron classifier.')
parser.add_argument('train_path', type=str, help='File path or URL to the training data')
parser.add_argument('test_path', type=str, help='File path or URL to the test data')
parser.add_argument('features', default='tweet_use', nargs='+', type=str, help='column name(s) of the features to use.')

parser.add_argument('--activation', default='relu', type=str, help='Activation function for the hidden layer.')
parser.add_argument('--solver', default='adam', type=str, help='The solver for weight optimization.')
parser.add_argument('--alpha', default=0.0001, type=float, help='L2 penalty (regularization term) parameter.')
parser.add_argument('--learning_rate', default='constant', type=str, help='Learning rate schedule for weight updates.')
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

# mlflow.sklearn.autolog()


def evaluate_multiclass(pipe, x_test, y_true, y_pred):
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


def evaluate_multilabel():
    pass


def main():
    with mlflow.start_run():
        logging.info("Loading training and testing sets, and feature sets")
        train_df, test_df = load(parsed_args.train_path, parsed_args.test_path)
        x_train = get_train_features(train_df, parsed_args.features)
        mlb, y_train, y_test = binarize_labels(y_train=train_df.label.tolist(),
                                               y_test=test_df.label.tolist())

        logging.info("Training mlp ...")
        mlp = MLPClassifier(**TRAIN_PARAMS)
        mlp.fit(x_train, y_train)
        mlp_model = LookupClassifier(mlb, mlp, parsed_args.features)

        logging.info("Predicting test ...")
        test_pred_df = predict_test(mlp_model, test_df)

        logging.info("Saving model ...")
        save_model(x_train, test_pred_df, mlp_model, ARTIFACT_PATH, REGISTERED_MODEL_NAME)

        # logging.info("Evaluating model ...")

        # metrics = evaluate_multiclass(mlp_model, x_test, y_test, [list(row) for row in y_pred_df.iterrows()])
        # mlflow.log_metrics(metrics)

        logging.info("Logging artifacts ...")
        log_artifacts(parsed_args.train_path, parsed_args.test_path)


if __name__ == '__main__':
    main()
