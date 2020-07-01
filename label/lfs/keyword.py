"""
References:
    https://www.snorkel.org/use-cases/01-spam-tutorial#a-keyword-lfs
    https://snorkel.readthedocs.io/en/v0.9.5/packages/_autosummary/labeling/snorkel.labeling.LabelingFunction.html
"""
from label.lfs import Label, KEYWORDS_YAML_FILENAME, ABSTAIN
from label.lfs.preprocessors import spacy_preprocessor
from snorkel.labeling import LabelingFunction, LFAnalysis
import en_core_web_sm
import yaml

nlp = en_core_web_sm.load()

with open(KEYWORDS_YAML_FILENAME, 'r') as infile:
    keyword_doc_dict = {}
    keyword_dict = yaml.load(infile, yaml.FullLoader)   # label: [keyword1, keyword2, ...]
    for category in keyword_dict:
        docs = list(nlp.pipe(keyword_dict[category]))   # [doc1, doc2, ...]
        keyword_doc_dict[category] = docs


def keyword_lookup(x, keyword_docs, label):
    doc_lemma_list = []
    x_lemma_list = [token.lemma_ if not token.is_stop else '' for token in x.spacy_doc]
    for doc in keyword_docs:
        doc_lemma_list = doc_lemma_list + [token.lemma_ for token in doc]
    return label if any(token in x_lemma_list for token in doc_lemma_list) else ABSTAIN


def make_keyword_lf(name, keyword_docs, label):
    return LabelingFunction(
        name=name,
        f=keyword_lookup,
        resources=dict(keyword_docs=keyword_docs, label=label),
        pre=[spacy_preprocessor]
    )


keyword_lfs = [make_keyword_lf('keyword_{0}'.format(label.name),
                               keyword_doc_dict[label.name],
                               label.value)
               for label in Label]


def evaluate_lfs(L_train):
    return LFAnalysis(L_train, lfs=keyword_lfs).lf_summary()
