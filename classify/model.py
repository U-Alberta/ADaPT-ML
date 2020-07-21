"""
References:
    https://keras.io/examples/imdb_bidirectional_lstm/
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highligh:logistic%20regression#sklearn.linear_model.LogisticRegression
"""
from label import DEMO_TRAINING_DATA_FILENAME
from classify import DEMO_FEATURIZER_FILENAME, DEMO_FEATURIZED_FILENAME, DEMO_MODEL_FILENAME, DEMO_BUNDLED_FILENAME
from utils.config import read_config
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle


def train_model(x_train: np.ndarray, train_df: pd.DataFrame, model_filename: str) -> LogisticRegression:
    train_params = read_config('mlogit')
    sklearn_model = LogisticRegression(**TRAIN_PARAMS)
    sklearn_model.fit(x_train, train_df.label.tolist())
    with open(model_filename, 'wb') as outfile:
        pickle.dump(sklearn_model, outfile)
    return sklearn_model


def create_bundle(featurizer: TfidfVectorizer, model: LogisticRegression, bundled_filename: str) -> ModelBundle:
    bundled = ModelBundle(featurizer, model)
    bundled.save(bundled_filename)
    return bundled


if __name__ == '__main__':
    with open(DEMO_FEATURIZED_FILENAME, 'rb') as infile:
        x_demo = pickle.load(infile)
    demo_train_df = pd.read_pickle(DEMO_TRAINING_DATA_FILENAME)
    demo_featurizer = pd.read_pickle(DEMO_FEATURIZER_FILENAME)
    demo_model = train_model(x_demo, demo_train_df, DEMO_MODEL_FILENAME)
    bundle = create_bundle(demo_featurizer, demo_model, DEMO_BUNDLED_FILENAME)
    eg = "I love Canadian oil and gas!"
    eg_featurized = bundle.featurize([eg])
    eg_predicted = bundle.predict(eg_featurized)
    print("Here's a new data point:\n{0}".format(eg))
    print("Here's the point featurized:\n", eg_featurized)
    print("Here's the point classified:\n", eg_predicted)
