import argparse
from ls import DATABASE_IP, LS_TASKS_PATH, CLASSIFICATION_TASKS
import pandas as pd
import json
from datetime import datetime
import os
import logging

date_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

parser = argparse.ArgumentParser(description='Sample a number of data points from a table to annotate.')
parser.add_argument('table', type=str, help='Table name that stores the data points.')
parser.add_argument('columns', nargs='+', type=str, help='column name(s) of the data point fields to use for '
                                                         'annotation.')
parser.add_argument('n', type=int, help='Number of data points to sample.')
parser.add_argument('task', type=str, choices=CLASSIFICATION_TASKS, help='What classification task is this '
                                                                         'sample for?')
parser.add_argument('--filename', default=None, type=str, help='What would you like the task file to be called?')
parsed_args = parser.parse_args()

FILENAME = parsed_args.filename if parsed_args.filename is not None else "{d}_{t}.json".format(d=date_str,
                                                                                               t=parsed_args.task)
JSON_TASKS_PATH = os.path.join(LS_TASKS_PATH, FILENAME)
LOGGING_FILENAME = os.path.join(LS_TASKS_PATH, '{}_log.txt'.format(FILENAME))
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=LOGGING_FILENAME, filemode='w')


def main():
    random_query = """
        SELECT {columns} FROM {table} 
        ORDER BY RANDOM() 
        LIMIT {n};""".format(columns=', '.join(['id'] + parsed_args.columns), table=parsed_args.table, n=parsed_args.n)
    tasks_df = pd.read_sql(random_query, DATABASE_IP)

    logging.info("Successfully sampled {n} data points from {table}".format(n=parsed_args.n, table=parsed_args.table))
    tasks_df['table_name'] = parsed_args.table

    task_json = []
    for row in tasks_df.itertuples(index=False):
        data_dict = {"ref_id": row.id, "meta_info": {"table_name": row.table_name, "task": parsed_args.task}}
        data_dict.update(dict(zip(parsed_args.columns, [getattr(row, c) for c in parsed_args.columns])))
        task_json.append(
            {
                "id": row.id,
                "data": data_dict
            }
        )

    with open(JSON_TASKS_PATH, 'w') as outfile:
        json.dump(task_json, outfile)

    logging.info("Saved {task} tasks in {path}".format(task=parsed_args.task, path=JSON_TASKS_PATH))


if __name__ == '__main__':
    main()
