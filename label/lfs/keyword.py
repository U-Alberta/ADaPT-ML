"""
References:
    https://www.snorkel.org/use-cases/01-spam-tutorial#a-keyword-lfs
"""
from label.lfs import Label, KEYWORDS_YAML_FILENAME
import yaml

from snorkel.labeling import LabelingFunction

with open(KEYWORDS_YAML_FILENAME, 'r') as infile:
    keyword_dict = yaml.load(infile, yaml.FullLoader)


def keyword_lookup(x, keywords, label):
    return label if any(word in x.text.lower() for word in keywords) else Label.ABSTAIN.value


def make_keyword_lf(keywords, label):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label)
    )


keywords_stp = make_keyword_lf(keywords=keyword_dict[Label.ST_P.name], label=Label.ST_P.value)
