"""
https://docs.python.org/3/library/abc.html
"""
import tensorflow as tf
from abc import ABC, abstractmethod
import pickle

from redditscore.tokenizer import CrazyTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


word_tokenizer = CrazyTokenizer(lowercase=True,
                                keepcaps=False,
                                ignore_stopwords=False,
                                decontract=True,
                                stem='lemm',
                                # extra_patterns=extra_patterns,
                                hashtags='split',
                                urls='',
                                whitespaces_to_underscores=False)

phrase_tokenizer = CrazyTokenizer(lowercase=False,
                                  keepcaps=True,
                                  hashtags='split',
                                  decontract=True,
                                  twitter_handles='',
                                  urls='',
                                  whitespaces_to_underscores=False)

class BaseModelBundle(ABC):
    """
    This base model is designed to have functions for preprocessing, featurizing, and modeling bundled into one
    tensorflow servable object
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, raw_docs: [str]):
        pass

    @abstractmethod
    def train(self, docs, y_train: [str]):
        pass

    @abstractmethod
    def featurize(self, processed_docs):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def save(self, bundled_filename):
        pass


class MlogitModelBundle(BaseModelBundle, tf.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.tokenizer = word_tokenizer
        self.featurizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        self.model = LogisticRegression(**self.params)

    @tf.function
    def preprocess(self, raw_docs: [str]):
        return [' '.join(self.tokenizer.tokenize(doc)) for doc in raw_docs]

    @tf.function
    def featurize(self, docs: [str]):
        return self.featurizer.transform(docs)

    @tf.function
    def train(self, raw_docs: [str], y: [str]):
        docs = self.preprocess(raw_docs)
        x_train = self.featurizer.fit_transform(docs)
        self.model.fit(x_train, y)
        return x_train

    def predict(self, raw_docs: [str]):
        docs = self.preprocess(raw_docs)
        x = self.featurize(docs)
        return x, self.model.predict(x)

    def save(self, bundled_filename):
        with open(bundled_filename, 'wb') as outfile:
            pickle.dump(self, outfile)


class RobertaLSTMModelBundle(BaseModelBundle, tf.keras.Model):

    def __init__(self, params):
        super().__init__()
        self.params = params
