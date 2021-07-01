from enum import Enum
from snorkel.labeling import LabelingFunction as LF
import inspect
import pickle

ABSTAIN = -1


class LabelingFunction(LF):

    def __init__(self, name, f):
        super(LabelingFunction, self).__init__(name=name, f=f)
        self.source = inspect.getsource(f)

    def __attrs(self):
        return self.name, self.source

    def __eq__(self, other):
        return isinstance(other, LabelingFunction) and self.__attrs() == other.__attrs()


class Example(Enum):
    cats = 0
    C = 0
    dogs = 1
    D = 1
    birds = 2
    B = 2


def load_lfs(saved_lfs_path, current_lfs):
    """
    This function runs a few checks to see if the saved LFs are up-to-date, i.e. they match all of the functions
    included in this module.
    """
    try:
        with open(saved_lfs_path, 'rb') as infile:
            saved_lfs = sorted(pickle.load(infile), key=lambda f: f.name)
        # Check that there are no new functions by name
        assert saved_lfs == sorted(current_lfs, key=lambda f: f.name)
        # Check that all of the existing functions have not been updated
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