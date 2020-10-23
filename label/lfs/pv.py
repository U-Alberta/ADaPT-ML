"""
References:
    https://www.snorkel.org/use-cases/01-spam-tutorial#a-keyword-lfs
    https://snorkel.readthedocs.io/en/v0.9.5/packages/_autosummary/labeling/snorkel.labeling.LabelingFunction.html
"""
import sqlite3

import en_core_web_lg
from snorkel.labeling import LabelingFunction, LFAnalysis

from label import LEXICONS_PATH
from label.lfs import ValueLabel, ABSTAIN
from label.lfs.preprocessors import spacy_preprocessor

nlp = en_core_web_lg.load()


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
    conn = sqlite3.connect(LEXICONS_PATH)
    cur = conn.cursor()
    value_doc_dict = {}
    for i in range(10):
        select_statement = """SELECT term FROM value_dict WHERE value = {0};""".format(i)
        cur.execute(select_statement)
        terms = [item[0] for item in cur.fetchall()]
        term_docs = list(nlp.pipe(terms))
        value_doc_dict[ValueLabel(i).name] = term_docs
    return value_doc_dict


def lemma_keyword_lookup(x, keyword_docs, label):
    """
    This function looks for all of the word lemmas for a given PV in the given data point x.
    :param x: data point
    :param keyword_docs: docs for the given PV
    :param label: the label for the given PV
    :return:
    """
    doc_lemma_list = []
    x_lemma_list = [token.lemma_ if not token.is_stop else '' for token in x.spacy_doc]
    for doc in keyword_docs:
        doc_lemma_list = doc_lemma_list + [token.lemma_ for token in doc]
    return label if any(token in x_lemma_list for token in doc_lemma_list) else ABSTAIN


def make_keyword_lf(name, keyword_docs, label):
    """
    This function makes all of the keyword lfs by calling the lemma_keyword_lookup helper and attaches the SpaCy
    preprocessor.
    :param name: the prefix 'keywords_' followed by the PV abbreviation
    :param keyword_docs: the SpaCy docs for the given PV
    :param label: the integer value for the given PV
    :return: a keyword-based LabelingFunction for a given PV
    """
    return LabelingFunction(
        name=name,
        f=lemma_keyword_lookup,
        resources=dict(keyword_docs=keyword_docs, label=label),
        pre=[spacy_preprocessor]
    )


def evaluate_keyword_lfs(L_train):
    """
    Computes the correlation, conflict, coverage, polarity, etc. for the LFs based on the PV dictionary
    :param L_train: label matrix for the train set
    :return:
    """
    return LFAnalysis(L_train, lfs=keyword_lfs).lf_summary()


keyword_dict = load_keyword_dictionary()
keyword_lfs = [make_keyword_lf('keywords_{0}'.format(label.name),
                               keyword_dict[label.name],
                               label.value)
               for label in ValueLabel]
