from label import CONFUSION_MATRIX_FILENAME, LF_SUMMARY_FILENAME
from snorkel.labeling import LFAnalysis
from label.lfs import ValueLabel
from label.lfs.pv import keyword_lfs
from snorkel.labeling.model import MajorityLabelVoter
from label.model import INIT_PARAMS
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import json

import pandas as pd

METRICS = ['accuracy', 'coverage', 'f1_micro', 'f1_macro']


def lf_summary(L_train, label_model):
    summary = LFAnalysis(L=L_train, lfs=keyword_lfs).lf_summary(est_weights=label_model.get_weights())
    summary.to_html(LF_SUMMARY_FILENAME)


def label_model_summary(L_train, label_model, y):
    lm_metrics = label_model.score(L_train, y, METRICS)
    c = confusion_matrix(y, label_model.predict(L_train), labels=[label.value for label in ValueLabel])
    plt.matshow(c)
    plt.colorbar()
    plt.savefig(CONFUSION_MATRIX_FILENAME, format='jpg')
    return lm_metrics
