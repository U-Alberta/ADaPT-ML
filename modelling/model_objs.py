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

    def __init__(self, mlb, classifier, features, used_inverse_labels=False):
        self.mlb = mlb
        self.classifier = classifier
        self.classes = mlb.classes_
        self.features = features
        self.conda_env = LOOKUP_CLASSIFIER_CONDA_ENV
        self.query = SQL_QUERY
        self.db = DATABASE_IP
        self.used_inverse_labels = used_inverse_labels

    def get_features(self, id_df):
        features_df = pd.read_sql(self.query.format(column=', '.join(self.features),
                                                    table=id_df.at[0, 'table'],
                                                    ids=str(tuple(id_df.id.tolist()))), self.db)
        feature_arrays = [np.array(features_df[feature].tolist()) for feature in features_df]
        try:
            x = np.concatenate(feature_arrays, axis=1)
        except np.AxisError:
            x = feature_arrays[0]
        return x

    def predict(self, id_df):
        """
        predict needs a dataframe with just the table name and id of the tweet. Given this info, it will look for the
        appropriate features and predict on those.
        """
        x = self.get_features(id_df)
        preds = self.classifier.predict(x)
        if self.used_inverse_labels:
            preds = self.mlb.transform(list(map(lambda p: [p], preds)))
        result_df = pd.merge(id_df, pd.DataFrame(preds, columns=self.classes), left_index=True, right_index=True)
        return result_df


def save_model(x_train, test_pred_df, model, artifact_path, registered_model_name):
    # signature = infer_signature(x_train, test_pred_df)
    input_example = x_train[:5]
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=model,
        conda_env=model.conda_env,
        registered_model_name=registered_model_name,
        # signature=signature,
        input_example=input_example
    )
