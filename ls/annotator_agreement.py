"""https://en.wikipedia.org/wiki/Krippendorff%27s_alpha The minimum acceptable alpha coefficient should be chosen
according to the importance of the conclusions to be drawn from imperfect data. When the costs of mistaken
conclusions are high, the minimum alpha needs to be set high as well. In the absence of knowledge of the risks of
drawing false conclusions from unreliable data, social scientists commonly rely on data with reliabilities α ≥ 0.800,
consider data with 0.800 > α ≥ 0.667 only to draw tentative conclusions, and discard data whose agreement measures α
< 0.667. """

RELIABLE = 0.8
UNRELIABLE = 0.667


from glob import glob
import argparse
import json
import os
import pandas as pd
import logging
from ls import LABEL_STUDIO_DIRECTORY
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import krippendorff
from functools import reduce


def get_dev_df(completions_dir):
    completion_files = glob(completions_dir)
    dev_df = pd.DataFrame()
    for filename in completion_files:
        with open(filename, 'r') as infile:
            completion = json.load(infile)

        df = pd.json_normalize(completion)
        try:
            gold_label = df.at[0, 'completions'][0]['result'][0]['value']['choices']
            if "NONE" not in gold_label:
                df = df.rename(columns={'id': 'file_id', 'data.ref_id': 'id', 'data.meta_info.table': 'table'})
                df['gold_label'] = [gold_label]
                dev_df = dev_df.append(df, ignore_index=True)
        except IndexError:
            logging.warning("This completion has no result?: {}".format(df.at[0, 'completions']))

    return dev_df


def calc_krippendorff_alpha(dfs):
    gold_label_cols_df = dfs[[col for col in dfs.columns if 'gold_label' in col]]
    gold_labels_combined = [' '.join(labels) for labels in [
        [val for sublist in label for val in sublist] for label in gold_label_cols_df.itertuples(index=False)]
                            ]
    vectorizer = CountVectorizer()
    label_count_matrix = vectorizer.fit_transform(gold_labels_combined)
    print("This is the vocabulary:")
    print(vectorizer.vocabulary)

    print("This is the label count matrix:")
    print(label_count_matrix.toarray())
    nominal_metric = krippendorff.alpha(value_counts=label_count_matrix, level_of_measurement='nominal')
    return nominal_metric


def main():
    parser = argparse.ArgumentParser(description='Compute the inter-annotator agreement for a given task.')
    parser.add_argument('task', type=str, help='What task has the annotated data? {pv}')
    parsed_args = parser.parse_args()

    project_directories = glob(os.path.join(LABEL_STUDIO_DIRECTORY, '{}_project_*'.format(parsed_args.task)))
    assert project_directories

    df_list = []
    for d in project_directories:
        completions_directory = os.path.join(d, 'completions', '*')
        df = get_dev_df(completions_directory).drop(columns=['completions', 'file_id', 'table', 'data.tweet'])
        suffix = '_{}'.format(d[-1])
        df = df.add_suffix(suffix).rename(columns={'id{}'.format(suffix): 'id'})
        df_list.append(df)
    dfs = reduce(lambda left, right: pd.merge(left, right, on=['id'], how='outer'), df_list).dropna()
    print("Annotations loaded. Here's a preview:")
    print(dfs)
    nominal_alpha = calc_krippendorff_alpha(dfs)
    logging.info("Nominal Alpha: {}".format(nominal_alpha))
    print("Nominal Alpha: {}".format(nominal_alpha))
    if nominal_alpha >= RELIABLE:
        print("Feel free to use these annotations as a benchmark.")
    elif nominal_alpha < UNRELIABLE:
        print("Discard these annotations and start again.")
    else:
        print("Use these annotations to make tentative conclusions only.")


if __name__ == '__main__':
    main()
