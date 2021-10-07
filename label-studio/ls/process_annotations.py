import argparse
import sys

import pandas as pd
from ls import LS_ANNOTATIONS_PATH, LABEL_PREFIX, CLASSIFICATION_TASKS
import os
import json
import logging
import random
import statistics

TASK = None
RANDOM = 'random'
MAJORITY = 'majority'
DROP = 'drop'
NONE = {'NONE', 'N/A'}


def create_annotations_df(filename):
    """
    Create a DataFrame with all of the available data from the exported annotations file.
    :param filename: str path to exported annotation JSON file
    :return: DataFrame with columns id, table, task, worker_1 ... worker_n
    """
    with open(os.path.join(LS_ANNOTATIONS_PATH, filename), 'r') as infile:
        ann_json = json.load(infile)
    df = pd.json_normalize(ann_json, sep='_').drop_duplicates(subset='data_ref_id', ignore_index=True)

    ann_df = pd.DataFrame()

    for row in df.itertuples(index=False):
        row_dict = {'id': [row.data_ref_id], 'table': [row.data_meta_info_table], 'task': [row.data_meta_info_task]}
        row_dict.update(dict(zip(
            ['{0}{1}'.format(LABEL_PREFIX, d['completed_by']['id']) for d in row.annotations],
            [[set(d['result'][0]['value']['choices'])] if not
             set(d['result'][0]['value']['choices']).issubset(NONE)
             else [None]
             for d in row.annotations]
        )))
        ann_df = ann_df.append(pd.DataFrame(row_dict), ignore_index=True)
    logging.info("Annotation df completed. Shape: {}. Here is a peek:".format(ann_df.shape))
    logging.info(ann_df.head())

    return ann_df


def create_tasks_df(ann_df):
    """
    drop annotators who didn't annotate this data, then drop data points with at least one annotator missing. Used
    to calculate Krippendorff's alpha.
    """
    tasks_df = ann_df.dropna(axis='columns', how='all').dropna()
    logging.info("Tasks df completed. Shape: {}. Here is a peek:".format(tasks_df.shape))
    logging.info(tasks_df.head())
    return tasks_df


def create_gold_df(task_df, gold_choice):
    """
    from the fully labeled data points in task_df, decide how to combine the annotations from multiple workers into
    one column of gold labels
    """
    try:
        worker = '{}{}'.format(LABEL_PREFIX, int(gold_choice))
        try:
            task_df['gold_label'] = task_df[worker]
        except Exception as err:
            logging.error("Could not use {}'s annotations as gold labels:\n{}\n".format(worker, err.args))
    except ValueError as err:
        labels_df = task_df[[col for col in task_df.columns if LABEL_PREFIX in col]]
        if gold_choice == RANDOM:
            task_df['gold_label'] = labels_df.apply(random_gold_choice, axis='columns')
        elif gold_choice == MAJORITY:
            task_df['gold_label'] = labels_df.apply(majority_gold_choice, axis='columns')
        elif gold_choice == DROP:
            task_df['gold_label'] = labels_df.apply(drop_gold_choice, axis='columns')
        else:
            logging.error("Could not create gold df:\n{}\n".format(err.args))
    task_df = task_df.dropna()
    logging.info("Gold dataframe created. Shape: {} Here's a peek:".format(task_df.shape))
    logging.info(task_df.head())
    return task_df


def random_gold_choice(worker_labels_series):
    """
    Choose the label that the majority of workers agree on. If they are evenly split, choose a worker's label randomly.
    :param worker_labels_series: pd.Series with index labels worker_1 ... worker_n
    :return: either the label that all workers agree on, or the label of the majority, or the label of a randomly chosen
    worker if they are all evenly split
    """
    agreed_label = get_agreed_label(worker_labels_series)
    if agreed_label is not None:
        return agreed_label
    else:
        majority = get_majority_label(worker_labels_series)
        return majority if majority is not None else worker_labels_series[random.choice(worker_labels_series.index)]


def majority_gold_choice(worker_labels_series):
    """
    Choose the label that the majority of workers agree on. If they are evenly split, drop that datapoint.
    :param worker_labels_series: pd.Series with index labels worker_1 ... worker_n
    :return: either the label that all workers agree on, or the label of the majority, or None if they are all evenly split
    """
    agreed_label = get_agreed_label(worker_labels_series)
    return agreed_label if agreed_label is not None else get_majority_label(worker_labels_series)


def drop_gold_choice(worker_labels_series):
    """
    Choose the label that the majority of workers agree on. If they are evenly split, drop that datapoint.
    :param worker_labels_series: pd.Series with index labels worker_1 ... worker_n
    :return: either the label that all workers agree on, or None if they disagree
    """
    return get_agreed_label(worker_labels_series)


def get_agreed_label(worker_labels_series):
    return worker_labels_series[
        random.choice(worker_labels_series.index)] if worker_labels_series.duplicated(keep=False).all() else None


def get_majority_label(worker_labels_series):
    multimode = statistics.multimode([label for sublist in worker_labels_series for label in sublist])
    return set(multimode) if sorted(multimode) != sorted(CLASSIFICATION_TASKS[TASK]) else None


def main():
    gold_choice_help = """How to settle disagreements between workers. id: Provide the id of the worker whose labels 
    will be chosen every time. {random}: The least strict. Choose the label that the majority of workers agree on. If 
    they are evenly split, choose a worker's label randomly. {majority}: More strict. Choose the label that the 
    majority of workers agree on. If they are evenly split, drop that datapoint. {drop}: The most strict. If workers 
    disagree at all, drop that datapoint.""".format(random=RANDOM,
                                                    majority=MAJORITY,
                                                    drop=DROP)
    parser = argparse.ArgumentParser(description='Format exported annotations into DataFrames ready for downstream '
                                                 'functions.')
    parser.add_argument('filename', type=str, help='Name of the exported annotations file.')
    parser.add_argument('task', type=str, choices=tuple(CLASSIFICATION_TASKS.keys()), help='Which task is the '
                                                                                           'annotations file for?')
    parser.add_argument('gold_choice', type=str, default=MAJORITY, help=gold_choice_help)
    parsed_args = parser.parse_args()

    global TASK
    TASK = parsed_args.task
    logging_filename = os.path.join(LS_ANNOTATIONS_PATH, TASK, 'process_log.txt')
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=logging_filename, filemode='w')

    ann_df = create_annotations_df(parsed_args.filename)
    try:
        assert len(ann_df.task.unique()) == 1

    except AssertionError as err:
        logging.error("This annotation file has multiple tasks...?\n{}\nStopping.".format(err.args))
        sys.exit(1)

    # save raw annotations df
    ann_df.to_pickle(os.path.join(LS_ANNOTATIONS_PATH, TASK, 'ann_df.pkl'))

    # create and save df used for annotator agreement
    task_df = create_tasks_df(ann_df)
    task_df.to_pickle(os.path.join(LS_ANNOTATIONS_PATH, TASK, 'task_df.pkl'))

    # create and save df used for development and validation
    gold_df = create_gold_df(task_df, parsed_args.gold_choice)
    gold_df.to_pickle(os.path.join(LS_ANNOTATIONS_PATH, TASK, 'gold_df.pkl'))


if __name__ == '__main__':
    main()
