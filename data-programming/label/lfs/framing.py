import logging
from snorkel.labeling import LabelingFunction
import pandas as pd
from label.lfs import FramingLabels, ABSTAIN
from label import DATABASE_IP
from scipy.spatial.distance import cdist

FRAME_ELEMENT_QUERY = """
SELECT * FROM frame_elements;
"""
TRLD = 0.5


def get_lfs(parsed_args) -> [LabelingFunction]:
    """
    This function creates and returns a list of all lfs in this module
    :return: A list of LabelingFunctions defined in this module
    """
    global TRLD
    TRLD = parsed_args.trld
    lfs = []
    element_lfs = []
    frame_elements_df = pd.read_sql(FRAME_ELEMENT_QUERY, DATABASE_IP)
    for label in FramingLabels:
        element_lfs = element_lfs + [make_element_lf(row.element_id, getattr(row, parsed_args.encoder),
                                                     'txt_clean_{}'.format(parsed_args.encoder), label)
                                     for row in frame_elements_df.itertuples(index=False)]
    lfs = lfs + element_lfs
    logging.info("LFs have been gathered.")
    return lfs


def frame_element_similarity(x, element, encoder, label) -> int:
    distances = cdist([element], x[encoder], 'cosine')[0]
    smallest = min(distances)
    similarity = 1 - smallest
    return label.value if similarity >= TRLD else ABSTAIN


def make_element_lf(element_id: str, element, encoder: str, label) -> LabelingFunction:
    return LabelingFunction(
        name=element_id,
        f=frame_element_similarity,
        resources=dict(element=element, encoder=encoder, label=label)
    )
