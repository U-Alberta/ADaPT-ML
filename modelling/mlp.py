"""
multi-layer perceptron classifier
"""
import argparse
import logging
import sys
import pickle

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature
from sklearn.metrics import plot_confusion_matrix, multilabel_confusion_matrix, f1_score

from modelling import (data, TEST_DF_FILENAME, TEST_DF_HTML_FILENAME,
                       CONFUSION_MATRIX_FILENAME, LOGGING_FILENAME, MLB_FILENAME)


parser = argparse.ArgumentParser(description='Train a multi-layer perceptron classifier.')
parser.add_argument('train_path', type=str, help='File path or URL to the training data')
parser.add_argument('test_path', type=str, help='File path or URL to the test data')

parser.add_argument('--activation', default='relu', type=str, help='Activation function for the hidden layer.')
parser.add_argument('--solver', default='adam', type=str, help='The solver for weight optimization.')
parser.add_argument('--alpha', default=0.0001, type=float, help='L2 penalty (regularization term) parameter.')
parser.add_argument('--learning_rate', default='constant', type=str, help='Learning rate schedule for weight updates.')
parser.add_argument('--learning_rate_init', default=0.001, type=float, help='The initial learning rate used. It controls the step-size in updating the weights.')
parser.add_argument('--power_t', default=0.5, type=float, help='The exponent for inverse scaling rate.')
parser.add_argument('--max_iter', default=200, type=int, help='Maximum number of iterations.')
parser.add_argument('--tol', default=1e-4, type=float, help='Tolerance for the optimization.')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for gradient descent update.')
parser.add_argument('--beta_1', default=0.9, type=float, help='Exponential decay rate for estimates of first moment vector in adam.')
parser.add_argument('--beta_2', default=0.999, type=float, help='Exponential decay rate for estimates of second moment vector in adam.')
parser.add_argument('--epsilon', default=1e-8, type=float, help='Value for numerical stability in adam.')
parser.add_argument('--n_iter_no_change', default=10, type=int, help='Maximum number of epochs to not meet tol improvement.')
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
            'beta_1': parsed_args.beta_1,                                               # keep this between 0 and 1
            'beta_2': parsed_args.beta_2,                                               # keep this between 0 and 1
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
            'momentum': parsed_args.momentum,                                           # keep this between 0 and 1
            'early_stopping': True,
            'n_iter_no_change': parsed_args.n_iter_no_change
        }
    }[TRAIN_PARAMS['solver']])

REGISTERED_MODEL_NAME = 'Tfidf_MLP'
# mlflow.sklearn.autolog()


class MLBPipe(mlflow.pyfunc.PythonModel):
    def __init__(self, mlb, pipe):
        self.mlb = mlb
        self.pipe = pipe

    def predict(self, model_input):
        return self.mlb.inverse_transform(self.pipe.predict(model_input.text.tolist()))


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

        train_df, test_df = data.load(parsed_args.train_path, parsed_args.test_path)

        x_train = train_df.text.to_frame()
        y_train = train_df.label.to_frame()
        x_train_list = train_df.text.tolist()
        y_train_list = train_df.label.tolist()

        x_test = test_df.text.to_frame()
        y_test = test_df.label.to_frame()
        x_test_list = test_df.text.tolist()
        y_test_list = y_test.label.tolist()

        try:
            assert isinstance(y_train_list[0], list)
            assert isinstance(y_test_list[0], list)
            is_multilabel = True
            mlb = MultiLabelBinarizer()
            y_train_list = mlb.fit_transform(y_train_list)
            y_test_list = mlb.transform(y_test_list)

            # with open(MLB_FILENAME, 'wb') as outfile:
            #     pickle.dump(mlb, outfile)
            # mlflow.log_artifact(MLB_FILENAME)
        except AssertionError:
            is_multilabel = False

        pipe = Pipeline([('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
                         ('mlp', MLPClassifier(**TRAIN_PARAMS))])

        logging.info("Transforming training data and training mlp ...")
        pipe.fit(x_train_list, y_train_list)
        mlb_pipe = MLBPipe(mlb, pipe)
        y_pred = mlb_pipe.predict(x_test)

        logging.info("Saving model ...")
        # signature = infer_signature(x_test, pd.DataFrame({'multilabels': y_pred}))
        input_example = x_train[:5]
        mlflow.pyfunc.log_model(
            artifact_path='tfidf_mlp',
            python_model=mlb_pipe,
            registered_model_name=REGISTERED_MODEL_NAME,
            # signature=signature,
            input_example=input_example
        )
        # mlflow.sklearn.log_model(
        #     pipe,
        #     'tfidf_mlp',
        #     registered_model_name=REGISTERED_MODEL_NAME,
        #     signature=signature,
        #     input_example=input_example)

        logging.info("Evaluating model ...")
        if is_multilabel:
            pass
        else:
            metrics = evaluate_multiclass(mlb_pipe, x_test_list, y_test_list, y_pred)
            mlflow.log_metrics(metrics)

        logging.info("Saving artifacts ...")
        test_df.insert(len(test_df.columns), 'pred', y_pred)
        pd.to_pickle(test_df, TEST_DF_FILENAME)
        test_df.to_html(TEST_DF_HTML_FILENAME)

        mlflow.log_artifact(parsed_args.train_path)
        mlflow.log_artifact(parsed_args.test_path)
        mlflow.log_artifact(TEST_DF_FILENAME)
        mlflow.log_artifact(TEST_DF_HTML_FILENAME)
        mlflow.log_artifact(CONFUSION_MATRIX_FILENAME)
        mlflow.log_artifact(LOGGING_FILENAME)


if __name__ == '__main__':
    main()
