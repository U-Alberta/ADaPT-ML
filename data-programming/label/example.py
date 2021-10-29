import os

from label import run, parser
from label.lfs import ExampleLabels
from label.lfs.example import get_lfs

REGISTERED_MODEL_NAME = 'ExampleLabelModel'
LF_FEATURES = {'txt_clean_lemma': None}
DEV_ANNOTATIONS_PATH = os.path.join('/annotations', 'example', 'gold_df.pkl')


def main():
    parsed_args = parser.parse_args()
    run.start(REGISTERED_MODEL_NAME, LF_FEATURES, DEV_ANNOTATIONS_PATH, get_lfs, ExampleLabels, parsed_args)


if __name__ == '__main__':
    main()
