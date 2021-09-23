import os
from sys import version_info
import mlflow
from mlflow.models import infer_signature
import pandas as pd
import numpy as np

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

LOOKUP_CLASSIFIER_CONDA_ENV = {
    'channels': ['defaults'],
    'dependencies': [
        'python={}'.format(PYTHON_VERSION),
        'pip',
        {
            'pip': [
                'mlflow',
                'crate',
                'SQLAlchemy',
                'sklearn',
                'numpy',
                'pandas'
            ],
        },
    ],
    'name': 'lookup_classifier_env'
}

DATABASE_IP = os.environ['DATABASE_IP']
SQL_QUERY = """
    SELECT {column} FROM {table} WHERE id IN {ids};
    """


class LookupClassifier(mlflow.pyfunc.PythonModel):

    def __init__(self, mlb, classifier, features, used_inverse_labels=False, can_predict_probs=True):
        self.mlb = mlb
        self.classifier = classifier
        self.classes = mlb.classes_
        self.num_classes = len(self.classes)
        self.features = features
        self.conda_env = LOOKUP_CLASSIFIER_CONDA_ENV
        self.query = SQL_QUERY
        self.db = DATABASE_IP
        self.used_inverse_labels = used_inverse_labels
        self.can_predict_probs = can_predict_probs
        self.prob_labels = ['prob_{}'.format(c) for c in self.classes]

    def get_features(self, id_df):
        features_df = pd.DataFrame()
        for tbl in id_df.table.unique():
            tbl_df = id_df.loc[(id_df.table == tbl)]
            id_list = tbl_df.id.tolist()
            ids = str(tuple(id_list)) if len(id_list) > 1 else "('{}')".format(id_list[0])
            tbl_f_df = pd.read_sql(self.query.format(column=', '.join(['id'] + self.features),
                                                     table=tbl,
                                                     ids=ids),
                                   self.db,
                                   chunksize=100)
            for data in tbl_f_df:
                features_df = features_df.append(data, ignore_index=True)
        id_f_df = pd.merge(id_df, features_df, on='id')
        feature_arrays = [np.array(id_f_df[feature].tolist()) for feature in self.features]
        try:
            x = np.concatenate(feature_arrays, axis=1)
        except np.AxisError:
            x = feature_arrays[0]
        return x

    def predict(self, id_df):
        """
        predict needs a dataframe with just the table name and id of the tweet. Given this info, it will look for the
        appropriate features and predict on those.
        self.classifier must have a method predict(x:{array-like, sparse matrix} of shape (n_samples, n_features))
        self.classifier may have a method predict_proba(X{array-like, sparse matrix} of shape (n_samples, n_features))
        """
        x = self.get_features(id_df)
        preds = self.classifier.predict(x)
        if self.used_inverse_labels:
            preds = self.mlb.transform(list(map(lambda p: [p], preds)))
        pred_df = pd.DataFrame(preds, columns=self.classes)
        pred_df['id'] = id_df['id']
        result_df = pd.merge(id_df, pred_df, on='id')
        if self.can_predict_probs:
            probs = self.classifier.predict_proba(x)
            prob_df = pd.DataFrame(probs, columns=self.prob_labels)
            prob_df['id'] = id_df['id']
            result_df = pd.merge(result_df, prob_df, on='id')
        return result_df

    def predict_to_csv(self, id_df):
        """
        return result df as a csv-formatted string
        """
        return self.predict(id_df).to_csv()
