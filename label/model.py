import sys

from snorkel.labeling.model import LabelModel
import numpy as np

from label import LABEL_MODEL_FILENAME
from label.lfs import Label


INIT_PARAMS = {
    'cardinality': len(Label),
    'verbose': True,
    'device': 'cpu'
}

TRAIN_PARAMS = {
    'n_epochs': 1000,
    'lr': 0.01,
    'l2': 0.0,
    'optimizer': 'sgd',
    'lr_scheduler': 'constant',
    'prec_init': 0.7,
    'log_freq': 100
}


def train_label_model(L_train: np.ndarray) -> LabelModel:
    label_model = LabelModel(**INIT_PARAMS)
    label_model.fit(L_train, **TRAIN_PARAMS)
    return label_model


def save_label_model(label_model: LabelModel):
    label_model.save(LABEL_MODEL_FILENAME)


def load_label_model():
    label_model = LabelModel(**INIT_PARAMS)
    try:
        label_model.load(LABEL_MODEL_FILENAME)
    except IOError:
        sys.exit("Label model not found. Please run Step 2 first.")
