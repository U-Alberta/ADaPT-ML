"""
multi-layer perceptron classifier
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
"""
import argparse
import logging
from math import floor, ceil

import matplotlib.pyplot as plt
import mlflow
import model.data as data
import model.tracking as tracking
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as eval
from model import (X_TRAIN_FILENAME, TRAIN_DF_FILENAME, TRAIN_DF_HTML_FILENAME, TEST_PRED_DF_FILENAME,
                   TEST_PRED_DF_HTML_FILENAME, CONFUSION_MATRIX_FILENAME, parser)
from model_objs import LookupClassifier
from sklearn.neural_network import MLPClassifier

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
parser.add_argument('--tol', default=0.0001, type=float, help='Tolerance for the optimization.')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for gradient descent update.')
parser.add_argument('--beta_1', default=0.9, type=float,
                    help='Exponential decay rate for estimates of first moment vector in adam.')
parser.add_argument('--beta_2', default=0.999, type=float,
                    help='Exponential decay rate for estimates of second moment vector in adam.')
parser.add_argument('--epsilon', default=0.00000001, type=float, help='Value for numerical stability in adam.')
parser.add_argument('--n_iter_no_change', default=10, type=int,
                    help='Maximum number of epochs to not meet tol improvement.')
parser.add_argument('--max_fun', default=15000, type=int, help='Maximum number of loss function calls.')
parser.add_argument('--random_state', default=0, type=int, help='Determines random number generation for weights and '
                                                                'bias initialization, train-test split if early '
                                                                'stopping is used, and batch sampling when '
                                                                'solver=’sgd’ or ‘adam’. Pass an int for reproducible '
                                                                'results across multiple function calls.')

parsed_args = parser.parse_args()

TRAIN_PARAMS = {
    'activation': parsed_args.activation,
    'solver': parsed_args.solver,
    'alpha': parsed_args.alpha,
    'max_iter': parsed_args.max_iter,
    'tol': parsed_args.tol,
    'verbose': True,
    'random_state': parsed_args.random_state if parsed_args.random_state else None
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


def common_metrics(y_true: np.ndarray, y_gold: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    functions that can be used for multiclass or multilabel.
    """
    metrics_dict = common_functions('LM', y_true, y_pred, y_prob)
    if y_gold is not None:
        metrics_dict.update(common_functions('GOLD', y_gold, y_pred, y_prob))
    return metrics_dict


def common_functions(eval_type, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        '{} accuracy'.format(eval_type): eval.accuracy_score(y_true, y_pred),
        '{} macro average precision score'.format(eval_type): eval.average_precision_score(y_true, y_prob),
        '{} F1 micro'.format(eval_type): float(eval.f1_score(y_true, y_pred, average='micro')),
        '{} F1 macro'.format(eval_type): float(eval.f1_score(y_true, y_pred, average='macro')),
        '{} F1 weighted'.format(eval_type): float(eval.f1_score(y_true, y_pred, average='weighted')),
        '{} hamming loss micro'.format(eval_type): eval.hamming_loss(y_true, y_pred),
        '{} hamming loss macro'.format(eval_type): eval.hamming_loss(y_true, y_pred),
        '{} hamming loss weighted'.format(eval_type): eval.hamming_loss(y_true, y_pred),
        '{} jaccard score micro'.format(eval_type): eval.jaccard_score(y_true, y_pred, average='micro'),
        '{} jaccard score macro'.format(eval_type): eval.jaccard_score(y_true, y_pred, average='macro'),
        '{} jaccard score weighted'.format(eval_type): eval.jaccard_score(y_true, y_pred, average='weighted')
    }


def binary_metrics(ravel_y_true: [str], ravel_y_gold: [str], ravel_y_pred: [str],
                   y_prob_pos: np.ndarray, pos_label: str) -> dict:
    metrics_dict = binary_functions('LM', ravel_y_true, ravel_y_pred, y_prob_pos, pos_label)
    if ravel_y_gold is not None:
        metrics_dict.update(binary_functions('GOLD', ravel_y_gold, ravel_y_pred, y_prob_pos, pos_label))
    return metrics_dict


def binary_functions(eval_type, ravel_y_true: [str], ravel_y_pred: [str],
                     y_prob_pos: np.ndarray, pos_label: str) -> dict:
    return {
        '{} DET curve'.format(eval_type): eval.det_curve(ravel_y_true, y_prob_pos,
                                                           pos_label=pos_label),
        '{} F1 binary'.format(eval_type): eval.f1_score(ravel_y_true, ravel_y_pred,
                                                          pos_label=pos_label),
        '{} hamming loss binary'.format(eval_type): eval.hamming_loss(ravel_y_true, ravel_y_pred,
                                                                        pos_label=pos_label),
        '{} jaccard score binary'.format(eval_type): eval.jaccard_score(ravel_y_true, ravel_y_pred,
                                                                          pos_label=pos_label)
    }


def evaluate_multiclass(y_true: np.ndarray,
                        y_gold: np.ndarray,
                        y_pred: np.ndarray,
                        y_prob: np.ndarray,
                        ravel_y_true: [str],
                        ravel_y_gold: [str],
                        ravel_y_pred: [str],
                        x_test: np.ndarray,
                        mlp_model: LookupClassifier) -> dict:
    metrics_dict = common_metrics(y_true, y_gold, y_pred, y_prob)
    metrics_dict.update(multiclass_functions('LM', ravel_y_true, ravel_y_pred))
    if ravel_y_gold is not None:
        metrics_dict.update(multiclass_functions('GOLD', ravel_y_gold, ravel_y_pred))
        eval.plot_confusion_matrix(mlp_model.classifier, x_test, ravel_y_gold, labels=mlp_model.classes)
    else:
        eval.plot_confusion_matrix(mlp_model.classifier, x_test, ravel_y_true, labels=mlp_model.classes)
    plt.savefig(CONFUSION_MATRIX_FILENAME)
    plt.close()
    return metrics_dict


def multiclass_functions(eval_type, ravel_y_true, ravel_y_pred):
    return {
        '{} balanced accuracy score'.format(eval_type): eval.balanced_accuracy_score(ravel_y_true, ravel_y_pred),
        '{} MCC'.format(eval_type): eval.matthews_corrcoef(ravel_y_true, ravel_y_pred)
    }


def evaluate_multilabel(y_true: np.ndarray,
                        y_gold: np.ndarray,
                        y_pred: np.ndarray,
                        y_prob: np.ndarray,
                        mlp_model: LookupClassifier) -> dict:
    metrics_dict = common_metrics(y_true, y_gold, y_pred, y_prob)
    metrics_dict.update(multilabel_functions('LM', y_true, y_prob))
    if y_gold is not None:
        metrics_dict.update(multilabel_functions('GOLD', y_gold, y_prob))
        # make multilabel confusion matrix with TPs in 1,1 and TN in 0,0
        cm = eval.multilabel_confusion_matrix(y_gold, y_pred)
    else:
        cm = eval.multilabel_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(floor(mlp_model.num_classes / 2), ceil(mlp_model.num_classes / 2), figsize=(12, 7))

    for axes, cfs_matrix, label in zip(ax.flatten(), cm, mlp_model.classes):
        plot_multilabel_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

    fig.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILENAME)
    plt.close()

    return metrics_dict


def multilabel_functions(eval_type, y_true, y_prob):
    return {
        '{} discounted cumulative gain'.format(eval_type): eval.dcg_score(y_true, y_prob)
    }


def plot_multilabel_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
    """
    Adapted from https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(class_label)


def main():
    with mlflow.start_run():
        run = mlflow.active_run()
        logging.info("Active run_id: {}".format(run.info.run_id))

        logging.info("Loading training and testing sets, and feature sets")
        train_df, test_df = data.load(parsed_args.train_path, parsed_args.test_path)
        data.save_df(train_df, TRAIN_DF_FILENAME, TRAIN_DF_HTML_FILENAME)
        x_train = data.get_train_features(train_df, parsed_args.features)
        data.save_training_features(x_train, X_TRAIN_FILENAME)

        logging.info("Encoding labels and determining multiclass or multilabel ...")
        try:
            mlb, y_train, y_test, y_gold = data.binarize_labels(y_train=train_df.label.tolist(),
                                                                y_test=test_df.label.tolist(),
                                                                y_gold=test_df.gold_label.tolist())
        except AttributeError as err:
            logging.info("This testing data has no gold label, so evaluation will be based on Label Model labels.")
            mlb, y_train, y_test, y_gold = data.binarize_labels(y_train=train_df.label.tolist(),
                                                                y_test=test_df.label.tolist())
        train_is_multiclass, test_is_multiclass = data.check_if_multiclass(y_train, y_test)

        logging.info("Training mlp ...")
        mlp = MLPClassifier(**TRAIN_PARAMS)
        if train_is_multiclass and test_is_multiclass:
            ravel_y_train = data.ravel_inverse_binarized_labels(mlb, y_train)
            mlp.fit(x_train, ravel_y_train)
            mlp_model = LookupClassifier(mlb, mlp, parsed_args.features, used_inverse_labels=True)
        else:
            mlp.fit(x_train, y_train)
            mlp_model = LookupClassifier(mlb, mlp, parsed_args.features)
        tracking.save_model(x_train, mlp_model, REGISTERED_MODEL_NAME, ARTIFACT_PATH)

        logging.info("Predicting test ...")
        test_pred_df = mlp_model.predict(test_df)
        data.save_df(test_pred_df, TEST_PRED_DF_FILENAME, TEST_PRED_DF_HTML_FILENAME)
        y_pred = test_pred_df[mlp_model.classes].to_numpy()
        y_prob = test_pred_df[mlp_model.prob_labels].to_numpy()
        logging.info("Evaluating model ...")
        try:
            if train_is_multiclass and test_is_multiclass:
                logging.info("Train and test are multiclass. Using multiclass evaluation ...")
                ravel_y_test = data.ravel_inverse_binarized_labels(mlb, y_test)
                try:
                    ravel_y_gold = data.ravel_inverse_binarized_labels(mlb, y_gold)
                except AttributeError:
                    ravel_y_gold = None
                ravel_y_pred = data.ravel_inverse_binarized_labels(mlb, y_pred)
                x_test = mlp_model.get_features(test_pred_df)
                metrics = evaluate_multiclass(y_test, y_gold, y_pred, y_prob,
                                              ravel_y_test, ravel_y_gold, ravel_y_pred,
                                              x_test, mlp_model)
                try:
                    # do binary classification metrics
                    pos_label = [label for label in mlp_model.prob_labels if '_pos' in label].pop()
                    y_prob_pos = test_pred_df[pos_label].to_numpy()
                    metrics.update(binary_metrics(ravel_y_test, ravel_y_gold, ravel_y_pred, y_prob_pos, pos_label))
                    logging.info("Binary classification metrics available.")
                except IndexError:
                    logging.info("The task is not binary, so no binary classification metrics are available.")
                except Exception as e:
                    logging.warning("Problem computing binary classification metrics:\n{}\n".format(e.args))

            elif not train_is_multiclass and not test_is_multiclass:
                logging.info("Train and test are multilabel. Using multilabel evaluation ...")
                metrics = evaluate_multilabel(y_test, y_gold, y_pred, y_prob, mlp_model)

            else:
                logging.warning("""Train and test do not have matching classification types. 
                This could have consequences in evaluation. Check data before continuing ...""")
                metrics = None
        except Exception as e:
            msg = "Unable to perform evaluation:\n{}\n".format(e.args)
            logging.error(msg)
            metrics = None

        logging.info("Logging artifacts ...")
        tracking.log(TRAIN_PARAMS, metrics)


if __name__ == '__main__':
    main()
