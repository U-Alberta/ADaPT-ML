"""
References:
    https://www.snorkel.org/use-cases/01-spam-tutorial#a-keyword-lfs
    https://snorkel.readthedocs.io/en/v0.9.5/packages/_autosummary/labeling/snorkel.labeling.LabelingFunction.html
"""
from label import LabelGroup
from label.lfs import ValueLabel, LEXICONS_PATH, ABSTAIN, KEYWORDS_YAML_FILENAME
from label.lfs.preprocessors import spacy_preprocessor
from snorkel.labeling import LabelingFunction, LFAnalysis
import en_core_web_lg
import sqlite3
import yaml

nlp = en_core_web_lg.load()


def load_keyword_dictionary():
    try:
        conn = sqlite3.connect(LEXICONS_PATH)
        cur = conn.cursor()
        value_doc_dict = {}
        for i in range(10):
            select_statement = """SELECT term FROM value_dict WHERE value = {0};""".format(i)
            cur.execute(select_statement)
            terms = cur.fetchall()
            term_docs = list(nlp.pipe(terms))
            value_doc_dict[ValueLabel(i).name] = term_docs
        return value_doc_dict
    except:
        with open(KEYWORDS_YAML_FILENAME, 'r') as infile:
            keyword_doc_dict = {}
            keyword_dict = yaml.load(infile, yaml.FullLoader)   # label: [keyword1, keyword2, ...]
            for category in keyword_dict:
                docs = list(nlp.pipe(keyword_dict[category]))   # [doc1, doc2, ...]
                keyword_doc_dict[category] = docs


def load_value_dictionary():
    conn = sqlite3.connect(LEXICONS_PATH)
    cur = conn.cursor()
    value_doc_dict = {}
    for i in range(10):
        select_statement = """SELECT term FROM value_dict WHERE value = {0};""".format(i)
        cur.execute(select_statement)
        terms = cur.fetchall()
        term_docs = list(nlp.pipe(terms))
        value_doc_dict[ValueLabel(i).name] = term_docs
    return value_doc_dict


def lemma_keyword_lookup(x, keyword_docs, label):
    doc_lemma_list = []
    x_lemma_list = [token.lemma_ if not token.is_stop else '' for token in x.spacy_doc]
    for doc in keyword_docs:
        doc_lemma_list = doc_lemma_list + [token.lemma_ for token in doc]
    return label if any(token in x_lemma_list for token in doc_lemma_list) else ABSTAIN


def make_keyword_lf(name, keyword_docs, label):
    return LabelingFunction(
        name=name,
        f=lemma_keyword_lookup,
        resources=dict(keyword_docs=keyword_docs, label=label),
        pre=[spacy_preprocessor]
    )


def make_keyword_lfs(keyword_doc_dict, label_group):
    return [make_keyword_lf('keywords_{0}'.format(label.name),
                            keyword_doc_dict[label.name],
                            label.value)
            for label in label_group]


def evaluate_keyword_lfs(L_train, lfs):
    return LFAnalysis(L_train, lfs=lfs).lf_summary()
