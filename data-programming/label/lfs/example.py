"""
Create a file like this one and write your LFs. As you update LFs or find new resources for new LFs, the load_lfs
function will make sure that you are staying up-to-date.
References:
    https://www.snorkel.org/use-cases/01-spam-tutorial#a-keyword-lfs
    https://snorkel.readthedocs.io/en/v0.9.5/packages/_autosummary/labeling/snorkel.labeling.LabelingFunction.html
"""
import logging
import os
from snorkel.labeling import LabelingFunction

from label.lfs import ExampleLabels, ABSTAIN

EXAMPLE_LFS_PATH = os.path.join('/lf_resources', 'example_lfs.pkl')
KEYWORD_DICT = {
    'cat': [
        'whisker', 'meow', 'litterbox', 'purr', 'hiss', 'lion', 'tiger', 'tabby',
        'leopard', 'panther', 'cheetah', 'cougar', 'kitten', 'cat'
    ],
    'dog': [
        'bark', 'pant', 'fetch', 'puppy', 'dog', 'beagle', 'collie', 'labrador',
        'schnauser', 'pitbull', 'bulldog', 'poodle', 'howl', 'husky'
    ],
    'bird': [
        'bird', 'chirp', 'squawk', 'wing', 'beak', 'feather', 'eagle', 'parrot',
        'vulture', 'toucan', 'talon', 'birdbath', 'penguin', 'chick'
    ]
}


def get_lfs():
    """
    This function creates a list of all lfs in this module
    """
    lfs = []
    keyword_lfs = []
    for label in ExampleLabels:
        keyword_lfs = keyword_lfs + [make_keyword_lf('keyword_{0}_{1}'.format(label.name, lemma),
                                                     lemma,
                                                     label)
                                     for lemma in KEYWORD_DICT[label.name]]
        lfs = lfs + keyword_lfs
    return lfs


def lemma_keyword_lookup(x, lemma, label):
    """
    This function looks for all of the word lemmas for a given label in the given data point x.
    :param x: data point
    :param keyword_docs: docs for the given label
    :param label: the label for the given label
    :return:
    """
    return label.value if lemma in x.text_lemm else ABSTAIN


def make_keyword_lf(name, lemma, label):
    """
    This function makes all of the keyword lfs by calling the lemma_keyword_lookup helper and attaches the SpaCy
    preprocessor.
    :param name: the prefix 'keywords_' followed by the label abbreviation
    :param keyword_doc: the SpaCy docs for the given label
    :param label: the integer value for the given label
    :return: a keyword-based LabelingFunction for a given label
    """
    return LabelingFunction(
        name=name,
        f=lemma_keyword_lookup,
        resources=dict(lemma=lemma, label=label)
    )
