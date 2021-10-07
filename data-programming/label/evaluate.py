"""
https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
"The coverage_error function computes the average number of labels that have to be included in the final prediction such
that all true labels are predicted. This is useful if you want to know how many top-scored-labels you have to predict in
average without missing any true one. The best value of this metrics is thus the average number of true labels."

"""
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from label import CONFUSION_MATRIX_FILENAME, LF_SUMMARY_DEV_FILENAME, LF_SUMMARY_TRAIN_FILENAME
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from snorkel.labeling import LFAnalysis

MULTICLASS_METRICS = {
    'binary': [
        'accuracy', 'coverage', 'precision', 'recall', 'f1', 'f1_micro', 'f1_macro', 'matthews_corrcoef',
        'roc_auc'
    ],
    'multiclass': [
        'coverage', 'f1_micro', 'f1_macro', 'matthews_corrcoef'
    ]
}


def lf_summary(train_L, dev_L, lfs, label_model, dev_true_lfs=None):
    """
    This function uses the train and dev label matrices, along with the LFs used to create them and the LF weights
    from the Label Model to calculate the coverage, conflict, empirical accuracy, and other metrics.
    :param train_L: NumPy matrix of ints corresponding to the votes from the LFs for the train data points
    :param dev_L: NumPy matrix of ints corresponding to the votes from the LFs for the dev data points
    :param lfs: [LabelingFunction] for the classification task curently in progress
    :param label_model: LabelModel that has been trained on train_L
    :param dev_true_lfs: [[label.value]] for the label class for the task in progress from Label Studio
    """
    try:
        # Try to get the empirical accuracy using the development set
        dev_summary = LFAnalysis(L=dev_L, lfs=lfs).lf_summary(Y=dev_true_lfs,
                                                              est_weights=label_model.get_weights())
        dev_summary.to_html(LF_SUMMARY_DEV_FILENAME)
    except Exception as err:
        if dev_L and dev_true_lfs:
            logging.warning("No empirical evaluation of LFs available:\n{}\n".format(err.args))
    train_summary = LFAnalysis(L=train_L, lfs=lfs).lf_summary(est_weights=label_model.get_weights())
    train_summary.to_html(LF_SUMMARY_TRAIN_FILENAME)


def multiclass_summary(train_L, dev_L, lfs, dev_true, dev_true_lfs, dev_pred, label_model) -> dict:
    """
    The multiclass summary is comprised of the confusion matrix for the label model's predicted labels against the
    gold labels, the accuracy, coverage, f1 micro, and f1 macro of the predictions on the development set.
    :param dev_L: NumPy matrix of ints corresponding to the votes from the LFs for the dev data points
    :param dev_true: [[label.name]] for the label class for the task in progress from Label Studio
    :param dev_pred: [[label.name]] for the label class for the task in progress from the label model
    :param label_model: LabelModel that has been trained on train_L
    :return: either None if no dev labels are available, or a dict for the metrics from the label model
    """
    try:
        class_labels = list(set(dev_true))
        dev_true_lfs = np.array(dev_true_lfs)
    except Exception as err:
        class_labels = None
        if dev_true and dev_true_lfs:
            logging.warning("Problem formatting labels:\n{}\n".format(err.args))
    lf_summary(train_L, dev_L, lfs, label_model, dev_true_lfs)
    try:
        cm = confusion_matrix(dev_true, dev_pred, labels=class_labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
        disp.plot()
        plt.savefig(CONFUSION_MATRIX_FILENAME, format='jpg')
    except Exception as err:
        if dev_true and dev_pred and class_labels:
            logging.warning("Confusion matrix not available:\n{}\n".format(err.args))
    try:
        if len(class_labels) == 2:
            lm_metrics = label_model.score(dev_L, dev_true_lfs, MULTICLASS_METRICS['binary'])
        else:
            lm_metrics = label_model.score(dev_L, dev_true_lfs, MULTICLASS_METRICS['multiclass'])
    except Exception as err:
        if class_labels and dev_L and dev_true_lfs:
            logging.warning("Label model metrics not available:\n{}\n".format(err.args))
        lm_metrics = None
    return lm_metrics


def multilabel_summary(train_L, dev_L, lfs, dev_true, dev_pred, label_model) -> dict:
    """
    For the multilabel summary, the confusion matrix is used to calculate performance metrics for the label model since
    the label model does not natively support a multilabel setting.
    :param dev_true: [[label.name]] for the label class for the task in progress from Label Studio
    :param dev_pred: [[label.name]] for the label class for the task in progress from the label model
    :return: either None if no dev labels are available, or a dict for the metrics from the label model
    """
    lf_summary(train_L, dev_L, lfs, label_model)
    mlb = MultiLabelBinarizer()
    try:
        dev_true = mlb.fit_transform(dev_true)
        dev_pred = mlb.transform(dev_pred)
        mcm = multilabel_confusion_matrix(dev_true, dev_pred)
        tn = sum(mcm[:, 0, 0])
        tp = sum(mcm[:, 1, 1])
        fn = sum(mcm[:, 1, 0])
        fp = sum(mcm[:, 0, 1])
        lm_metrics = {
            'accuracy': (tp + tn) / (tn + tp + fn + fp),
            'precision': tp / (tp + fp),
            'recall': tp / (tp + fn),
            'specificity': tn / (tn + fp),
            'fall out': fp / (fp + tn),
            'miss rate': fn / (fn + tp),
            'F1 weighted': f1_score(dev_true, dev_pred, average='weighted'),
            'F1 samples': f1_score(dev_true, dev_pred, average='samples')
        }
    except Exception as err:
        if dev_true and dev_pred:
            logging.warning("Summary will not be available:\n{}\n".format(err.args))
        lm_metrics = None
    return lm_metrics


def get_dev_df(gold_df_path) -> pd.DataFrame:
    """
    Load the gold DataFrame that has compiled all workers' labels into one gold label set.
    """
    gold_df = pd.read_pickle(gold_df_path)
    gold_df['gold_label'] = gold_df['gold_label'].apply(lambda x: list(x))
    logging.info("Gold df loaded. Here's a peek:")
    logging.info(gold_df.head())
    return gold_df
