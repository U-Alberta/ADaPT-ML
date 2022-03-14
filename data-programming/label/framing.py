import os

from label import run, parser
from label.lfs import FramingLabels
from label.lfs.framing import get_lfs

REGISTERED_MODEL_NAME = 'FramingLabelModel'
LF_FEATURES = {
    'txt_clean_roberta': None,
    'txt_clean_use': None,
    }
DEV_ANNOTATIONS_PATH = os.path.join('/annotations', 'framing', 'gold_df.pkl')


def main():
    parser.add_argument('--trld', default=0.5, type=float, help='cosine similarity threshold')
    parser.add_argument('--encoder', default='roberta', choices=('roberta', 'use'), type=str,
                        help='which encoder embeddings to use')
    parsed_args = parser.parse_args()
    run.start(REGISTERED_MODEL_NAME, LF_FEATURES, DEV_ANNOTATIONS_PATH, get_lfs, FramingLabels, parsed_args)


if __name__ == '__main__':
    main()
