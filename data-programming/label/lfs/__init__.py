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
    cats = 0
    C = 0
    dogs = 1
    D = 1
    birds = 2
    B = 2
