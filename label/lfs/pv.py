"""
References:
    https://www.snorkel.org/use-cases/01-spam-tutorial#a-keyword-lfs
    https://snorkel.readthedocs.io/en/v0.9.5/packages/_autosummary/labeling/snorkel.labeling.LabelingFunction.html
"""
import os
import sys
import logging
import pickle
from snorkel.labeling import LabelingFunction, LFAnalysis

from label.lfs import ValueLabel, ABSTAIN
import requests

PV_DICTIONARY_URL = os.environ['PERSONAL_VALUES_DICTIONARY']

PV_LFS_PATH = os.path.join('/lf_resources', 'lfs.pkl')


def load_lfs():
    try:
        with open(PV_LFS_PATH, 'rb') as infile:
            lfs = pickle.load(infile)
        # assert len([lf.name for lf in lfs if lf.name.startswith('keyword')]) == 1068
        logging.info("Using existing LFs.")
    except (FileNotFoundError, AssertionError, EOFError):
        # remake all of the lfs to get updated ones
        logging.info("Remaking keyword LFs ...")
        personal_values_dict = load_keyword_dictionary()
        keyword_lfs = []
        for label in ValueLabel:
            keyword_lfs = keyword_lfs + [make_keyword_lf('keyword_{0}_{1}'.format(label.name, lemma),
                                                         lemma,
                                                         label)
                                         for lemma in personal_values_dict[label.name]]
        lfs = keyword_lfs
        with open(PV_LFS_PATH, 'wb') as outfile:
            pickle.dump(lfs, outfile)
        logging.info("New LFs saved.")
    return lfs

    
def load_keyword_dictionary():
    """
    There are 1068 terms in the personal values (PV) dictionary.
        SE: 85
        CO: 129
        TR: 109
        BE: 95
        UN: 103
        SD: 140
        ST: 120
        HE: 97
        AC: 88
        PO: 102
    This function connects to the database where the dictionary is stored, fetches the words for each PV, creates a
    SpaCy doc for them, and returns a dict in the form of {V1: [doc, ...], V2: [doc, ...] ...}
    :return:    dict
    """
    # TODO: update this so it gets the lemmas in the personal values dictionary
    try:
        personal_values_dict = requests.get(PV_DICTIONARY_URL).json()
        assert sorted(personal_values_dict) == sorted([label.name for label in ValueLabel])
        return personal_values_dict
    except Exception as e:
        sys.exit(e.args)


def lemma_keyword_lookup(x, lemma, label):
    """
    This function looks for all of the word lemmas for a given PV in the given data point x.
    :param x: data point
    :param keyword_docs: docs for the given PV
    :param label: the label for the given PV
    :return:
    """
    return label.value if (label.name in x.tweet_pv_words_count
                           and lemma in x.tweet_pv_words_count[label.name]['words']) else ABSTAIN


def make_keyword_lf(name, lemma, label):
    """
    This function makes all of the keyword lfs by calling the lemma_keyword_lookup helper and attaches the SpaCy
    preprocessor.
    :param name: the prefix 'keywords_' followed by the PV abbreviation
    :param keyword_doc: the SpaCy docs for the given PV
    :param label: the integer value for the given PV
    :return: a keyword-based LabelingFunction for a given PV
    """
    return LabelingFunction(
        name=name,
        f=lemma_keyword_lookup,
        resources=dict(lemma=lemma, label=label)
    )


def evaluate_keyword_lfs(L_train):
    """
    Computes the correlation, conflict, coverage, polarity, etc. for the LFs based on the PV dictionary
    :param L_train: label matrix for the train set
    :return:
    """
    return LFAnalysis(L_train, lfs=pv_lfs).lf_summary()


pv_lfs = load_lfs()
