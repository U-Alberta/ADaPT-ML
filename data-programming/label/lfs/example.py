"""
This module contains examples for keyword labeling functions and a function that gathers them for the label matrix
creation. Labeling functions can be created by using the function decorator like:

>>> @labeling_function(name="my_lf")
... def g(x):
...     return 0 if x.a > 42 else -1
>>> g
LabelingFunction my_lf, Preprocessors: []

Or labeling functions can be created directly as done in this example.

Create a file like this one and write your LFs. As you update LFs or find new resources for new LFs, update the
get_lfs function to include those new LFs. If a human annotator was reading over a data point and found a word like
meow, bark, or chirp, they might say the text is about a cat, dog, or bird, respectively. Another resource could be
the embeddings of these keywords using different models to disambiguate the keywords (meow is quite unambiguous,
but bark could refer to tree bark and chirp could refer to a general high pitch sound). With multimodal data,
these LF resources could be a trained image classification model that can find cats, dogs, and birds in images.

References:
    https://www.snorkel.org/use-cases/01-spam-tutorial#a-keyword-lfs
    https://snorkel.readthedocs.io/en/v0.9.7/packages/_autosummary/labeling/snorkel.labeling.labeling_function.html
    https://snorkel.readthedocs.io/en/v0.9.7/packages/_autosummary/labeling/snorkel.labeling.LabelingFunction.html
"""
import logging
from snorkel.labeling import LabelingFunction

from label.lfs import ExampleLabels, ABSTAIN

# This is an example of a keyword resource for creating some LFs.
KEYWORD_DICT = {
    'cat': [
        'whisker', 'meow', 'litterbox', 'purr', 'hiss', 'lion', 'tiger', 'tabby',
        'leopard', 'panther', 'cheetah', 'cougar', 'kitten', 'cat'
    ],
    'dog': [
        'bark', 'pant', 'woof', 'puppy', 'dog', 'beagle', 'collie', 'labrador',
        'schnauser', 'pitbull', 'bulldog', 'poodle', 'howl', 'husky'
    ],
    'bird': [
        'bird', 'chirp', 'squawk', 'wing', 'beak', 'feather', 'eagle', 'parrot',
        'vulture', 'toucan', 'talon', 'birdbath', 'penguin', 'chick'
    ]
}


def get_lfs() -> [LabelingFunction]:
    """
    This function creates and returns a list of all lfs in this module
    :return: A list of LabelingFunctions defined in this module
    """
    lfs = []
    keyword_lfs = []
    # Create a LabelingFunction for each lemma in the KEYWORD_DICT that votes for the category it falls under
    for label in ExampleLabels:
        keyword_lfs = keyword_lfs + [make_keyword_lf('keyword_{0}_{1}'.format(label.name, lemma),
                                                     lemma,
                                                     label)
                                     for lemma in KEYWORD_DICT[label.name]]
        lfs = lfs + keyword_lfs
    logging.info("LFs have been gathered.")
    return lfs


def lemma_keyword_lookup(x, lemma, label) -> int:
    """
    This function looks for all of the word lemmas for a given label in the given data point x.
    :param x: data point
    :param lemma: the lemma that this labeling function will look for in the data point
    :param label: the attribute of the ExampleLabels for the given label, holding its name and value
    :return: the integer value of the ExampleLabel if the lemma is found, or -1 (abstain) if not
    """
    return label.value if lemma in x.text_lemm else ABSTAIN


def make_keyword_lf(name, lemma, label) -> LabelingFunction:
    """
    This function makes all of the keyword lfs
    :param name: the prefix 'keyword_' followed by the label name and the lemma within that label
    :param lemma: the the lemma that belongs to the label this LF is being created for
    :param label: the attribute of the ExampleLabels for the given label, holding its name and value
    :return: a keyword-based LabelingFunction for a given label
    """
    return LabelingFunction(
        name=name,
        f=lemma_keyword_lookup,
        resources=dict(lemma=lemma, label=label)
    )
