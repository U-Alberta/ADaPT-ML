"""
References:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
"""
from label import DEMO_TRAINING_DATA_FILENAME
from classify import DEMO_FEATURIZED_FILENAME, DEMO_FEATURIZER_FILENAME
import pandas as pd
import logging
import sys
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def load_train_df(train_df_filename: str):



def featurize_text(train_df: pd.DataFrame, featurizer_filename: str, train_featurized_filename: str) -> np.ndarray:
    logging.info("Featurizing text ...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    x_train = vectorizer.fit_transform(train_df.all_text.tolist())
    with open(featurizer_filename, 'wb') as outfile:
        pickle.dump(vectorizer, outfile)
    with open(train_featurized_filename, 'wb') as outfile:
        pickle.dump(x_train, outfile)
    return x_train


if __name__ == '__main__':
    demo_train_df = load_train_df(DEMO_TRAINING_DATA_FILENAME)
    x_demo = featurize_text(demo_train_df, DEMO_FEATURIZER_FILENAME, DEMO_FEATURIZED_FILENAME)
    with pd.option_context('display.max_colwidth', -1):
        print("Here's a peek at the demo DF:\n", demo_train_df.head())
        print("Here's a peek at the demo text featurized:\n", x_demo)
