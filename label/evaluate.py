"""
https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
"The coverage_error function computes the average number of labels that have to be included in the final prediction such
that all true labels are predicted. This is useful if you want to know how many top-scored-labels you have to predict in
average without missing any true one. The best value of this metrics is thus the average number of true labels."

"""
import json
import logging
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from snorkel.labeling import LFAnalysis

from label import CONFUSION_MATRIX_FILENAME, LF_SUMMARY_DEV_FILENAME, LF_SUMMARY_TRAIN_FILENAME, DEV_DF_FILENAME, \
    DEV_DF_HTML_FILENAME

MULTICLASS_METRICS = ['accuracy', 'coverage', 'f1_micro', 'f1_macro']


def lf_summary(train_L, dev_L, lfs, label_model, dev_true):
    try:
        dev_summary = LFAnalysis(L=dev_L, lfs=lfs).lf_summary(Y=dev_true, est_weights=label_model.get_weights())
        dev_summary.to_html(LF_SUMMARY_DEV_FILENAME)
    except:
        logging.warning("No empirical evaluation of LFs available.")
    train_summary = LFAnalysis(L=train_L, lfs=lfs).lf_summary(est_weights=label_model.get_weights())
    train_summary.to_html(LF_SUMMARY_TRAIN_FILENAME)


def multiclass_summary(dev_L, dev_true, dev_pred, label_model) -> dict:
    try:
        # flatten arrays for evaluation
        dev_true = np.array(dev_true).flatten()
        dev_pred = np.array(dev_pred).flatten()
    except:
        logging.warning("Summary will not be available.")
    try:
        c = confusion_matrix(dev_true, dev_pred)
        plt.matshow(c)
        plt.colorbar()
        plt.savefig(CONFUSION_MATRIX_FILENAME, format='jpg')
    except:
        logging.warning("Confusion matrix not available.")
    try:
        lm_metrics = label_model.score(dev_L, dev_true, MULTICLASS_METRICS)
    except:
        logging.warning("Label model metrics not available.")
        lm_metrics = None
    return lm_metrics


def multilabel_summary(dev_true, dev_pred) -> dict:
    mlb = MultiLabelBinarizer()
    try:
        dev_true = mlb.fit_transform(dev_true)
        dev_pred = mlb.transform(dev_pred)
        mcm = multilabel_confusion_matrix(dev_true, dev_pred)
        tn = mcm[:, 0, 0]
        tp = mcm[:, 1, 1]
        fn = mcm[:, 1, 0]
        fp = mcm[:, 0, 1]
        lm_metrics = {'recall': tp / (tp + fn),
                      'specificity': tn / (tn + fp),
                      'fall_out': fp / (fp + tn),
                      'miss_rate': fn / (fn + tp),
                      'f1_weighted': f1_score(dev_true, dev_pred, average='weighted'),
                      'f1_samples': f1_score(dev_true, dev_pred, average='samples')}
    except:
        logging.warning("Summary will not be available.")
        lm_metrics = None
    return lm_metrics


def get_dev_df(completions_dir):
    completion_files = glob(completions_dir)
    dev_df = pd.DataFrame()
    for filename in completion_files:
        with open(filename, 'r') as infile:
            completion = json.load(infile)

        df = pd.json_normalize(completion)
        gold_label = df.at[0, 'completions'][0]['result'][0]['value']['choices']
        if "NONE" not in gold_label:
            df = df.rename(columns={'id': 'file_id', 'data.ref_id': 'id', 'data.meta_info.table': 'table'})
            df['gold_label'] = [gold_label]
            dev_df = dev_df.append(df, ignore_index=True)

    dev_df.to_pickle(DEV_DF_FILENAME)
    dev_df.to_html(DEV_DF_HTML_FILENAME)
    return dev_df
