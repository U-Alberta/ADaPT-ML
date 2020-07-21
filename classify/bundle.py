"""
https://docs.python.org/3/library/abc.html
"""
from abc import ABC, abstractmethod
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.config import read_config


class BaseModelBundle(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, docs: [str], y: [str]):
        pass

    @abstractmethod
    def featurize(self, docs: [str]):
        pass

    @abstractmethod
    def predict(self, docs: [str]):
        pass

    @abstractmethod
    def save(self, bundled_filename):
        pass


class MlogitModelBundle(BaseModelBundle):

    def __init__(self):
        super().__init__()
        self.params = read_config('mlogit')
        self.featurizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        self.model = LogisticRegression(**self.params)

    def train(self, docs: [str], y: [str]):
        x_train = self.featurizer.fit_transform(docs)
        self.model.fit(x_train, y)
        return x_train

    def featurize(self, docs: [str]):
        return self.featurizer.transform(docs)

    def predict(self, docs: [str]) -> [str]:
        x_test = self.featurize(docs)
        return self.model.predict(x_test)

    def save(self, bundled_filename):
        with open(bundled_filename, 'wb') as outfile:
            pickle.dump(self, outfile)
