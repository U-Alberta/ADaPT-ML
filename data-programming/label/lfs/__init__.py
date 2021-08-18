"""
Initialize the classes that store the mapping between class labels and the number they will
represent in the label matrix.
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
