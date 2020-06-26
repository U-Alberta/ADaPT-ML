"""
References:
    https://www.snorkel.org/use-cases/01-spam-tutorial#a-keyword-lfs
"""
from label.lfs import Label, KEYWORDS_YAML_FILENAME, ABSTAIN
import yaml

from snorkel.labeling import LabelingFunction

with open(KEYWORDS_YAML_FILENAME, 'r') as infile:
    keyword_dict = yaml.load(infile, yaml.FullLoader)


def keyword_lookup(x, keywords, label):
    return label if any(word in x.text.lower() for word in keywords) else ABSTAIN


def make_keyword_lf(name, keywords, label):
    return LabelingFunction(
        name=name,
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label)
    )


keyword_lfs = [make_keyword_lf('keyword_'.format(label.name),
                               keyword_dict[label.name],
                               label.value)
               for label in Label]
