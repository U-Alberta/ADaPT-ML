import os

from label import run
from label.lfs import ExampleLabels
from label.lfs.example import get_lfs

REGISTERED_MODEL_NAME = 'ExampleLabelModel'
LF_FEATURES = {'txt_clean_lemma': None}
DEV_ANNOTATIONS_PATH = os.path.join('/annotations', 'example', 'gold_df.pkl')


def main():
    run.start(REGISTERED_MODEL_NAME, LF_FEATURES, DEV_ANNOTATIONS_PATH, get_lfs, ExampleLabels)


if __name__ == '__main__':
    main()
