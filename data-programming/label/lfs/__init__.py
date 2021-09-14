"""
Initialize the classes that store the mapping between class labels and the number they will
represent in the label matrix.

For binary classification tasks, indicate which label is the positive class and which label is the negative class --
this will make the labels compatible with the modelling evaluation functions.

E.g.
        class BinaryLabels(Enum):
            ham_neg = 0
            spam_pos = 1
"""
from enum import Enum

ABSTAIN = -1


class ExampleLabels(Enum):
    """
    This is an example of 3 different categories that the data can be classified into -- the topic of cats, dogs,
    and / or birds
    """
    cat = 0
    dog = 1
    bird = 2
    horse = 3
    snake = 4
