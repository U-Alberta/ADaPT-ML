import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from snorkel.labeling import LFAnalysis

from label import CONFUSION_MATRIX_FILENAME, LF_SUMMARY_FILENAME

METRICS = ['accuracy', 'coverage', 'f1_micro', 'f1_macro']


def lf_summary(L_train, lfs, label_model):
    summary = LFAnalysis(L=L_train, lfs=lfs).lf_summary(est_weights=label_model.get_weights())
    summary.to_html(LF_SUMMARY_FILENAME)


def multiclass_summary(L_train, y_dev, label_model, labels):
    lm_metrics = label_model.score(L_train, y_dev, METRICS)
    c = confusion_matrix(y_dev, label_model.predict(L_train), labels=[label.value for label in labels])
    plt.matshow(c)
    plt.colorbar()
    plt.savefig(CONFUSION_MATRIX_FILENAME, format='jpg')
    return lm_metrics


def multilabel_summary(labeled_train_df, label_model):
    pass


def get_dev_df():
    pass
