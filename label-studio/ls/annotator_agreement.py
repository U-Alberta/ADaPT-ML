"""https://en.wikipedia.org/wiki/Krippendorff%27s_alpha The minimum acceptable alpha coefficient should be chosen
according to the importance of the conclusions to be drawn from imperfect data. When the costs of mistaken
conclusions are high, the minimum alpha needs to be set high as well. In the absence of knowledge of the risks of
drawing false conclusions from unreliable data, social scientists commonly rely on data with reliabilities α ≥ 0.800,
consider data with 0.800 > α ≥ 0.667 only to draw tentative conclusions, and discard data whose agreement measures α
< 0.667. """

import argparse
import json
import logging
import os
import sys
from functools import reduce
from glob import glob

import krippendorff
import pandas as pd
from ls import LS_ANNOTATIONS_PATH
from sklearn.feature_extraction.text import CountVectorizer

RELIABLE = 0.8
UNRELIABLE = 0.667
LABEL_PREFIX = 'worker_'

LOGGING_FILENAME = os.path.join(LS_ANNOTATIONS_PATH, 'agreement_log.txt')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=LOGGING_FILENAME, filemode='w')


def create_annotations_df(filename):
    with open(os.path.join(LS_ANNOTATIONS_PATH, filename), 'r') as infile:
        ann_json = json.load(infile)
    df = pd.json_normalize(ann_json, sep='_').drop_duplicates(subset='data_ref_id', ignore_index=True)

    ann_df = pd.DataFrame()

    for row in df.itertuples(index=False):
        row_dict = {'id': [row.data_ref_id], 'task': [row.data_meta_info_task]}
        row_dict.update(dict(zip(
            ['{0}{1}'.format(LABEL_PREFIX, d['completed_by']['id']) for d in row.annotations],
            [[d['result'][0]['value']['choices']] for d in row.annotations]
        )))
        ann_df = ann_df.append(pd.DataFrame(row_dict), ignore_index=True)
    logging.info("Annotation df completed. Here is a peek:")
    logging.info(ann_df.head())

    return ann_df


def calc_krippendorff_alpha(df: pd.DataFrame) -> float:
    """
    This function calculates krippendorff's alpha to measure annotator agreement for both multiclass and multilabel
    settings with two or more annotators.
    :param df: pd.Dataframe for a specific task with columns id, task, and worker_# for all data points that have been
    labeled by all annotators.
    :return: the nominal alpha
    """
    labels_df = df[[col for col in df.columns if LABEL_PREFIX in col]]
    labels_combined = [' '.join(labels) for labels in [
        [val for sublist in label for val in sublist] for label in labels_df.itertuples(index=False)]
                       ]
    vectorizer = CountVectorizer()
    label_count_matrix = vectorizer.fit_transform(labels_combined).toarray()
    nominal_metric = krippendorff.alpha(value_counts=label_count_matrix, level_of_measurement='nominal')

    return nominal_metric


def report(task, alpha):
    """
    This function generates a plain text report of the resulting nominal alpha for the given task, and the
    interpretation based on this Wikipedia article section:
    https://en.wikipedia.org/wiki/Krippendorff%27s_alpha#Significance
    :param task: The classification task these data points correspond to
    :param alpha: the nominal alpha
    """
    if alpha >= RELIABLE:
        result = "{0} >= {1}. Feel free to use these annotations as a benchmark.".format(alpha, RELIABLE)
    elif alpha < UNRELIABLE:
        result = "{0} < {1}. Discard these annotations and start again.".format(alpha, UNRELIABLE)
    else:
        result = "{0} <= {1} < {2}. Use these annotations to make tentative conclusions only.".format(UNRELIABLE,
                                                                                                      alpha, RELIABLE)
    summary_str = """
    TASK: {task}
    NOMINAL ALPHA: {alpha}
    RESULT: {result} 
    """.format(task=task, alpha=alpha, result=result)
    logging.info(summary_str)
    print(summary_str)


def main():
    parser = argparse.ArgumentParser(description='Compute the inter-annotator agreement for completed annotations.')
    parser.add_argument('filename', type=str, help='Name of the exported annotations file to calculate agreement for.')
    parsed_args = parser.parse_args()

    try:
        ann_df = create_annotations_df(parsed_args.filename)
    except Exception as e:
        logging.error("There was an issue creating the DataFrame:\n{}\nStopping.".format(e.args))
        sys.exit(1)

    for task in ann_df.task.unique():
        try:
            # drop annotators who didn't annotate this data, then drop data points with at least one annotator missing
            task_df = ann_df[(ann_df.task == task)].dropna(axis='columns', how='all').dropna()
            nominal_alpha = calc_krippendorff_alpha(task_df)
            report(task, nominal_alpha)
        except Exception as e:
            logging.error("Could not generate report for {0}:\n{1}\nStopping.".format(task, e.args))
            sys.exit(1)


if __name__ == '__main__':
    main()
