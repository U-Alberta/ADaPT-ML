import argparse
from ls import DATABASE_IP, LS_TASKS_PATH
import pandas as pd
import json
from datetime import datetime
import os
import logging


date_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = date_str+'.json'


parser = argparse.ArgumentParser(description='Sample a number of data points from a table to annotate.')
parser.add_argument('table', type=str, help='Table name that stores the data points.')
parser.add_argument('columns', nargs='+', type=str, help='column name(s) of the data point fields to use for '
                                                         'annotation.')
parser.add_argument('n', type=int, help='Number of data points to sample.')
parser.add_argument('task', type=str, default='example', choices=('example',), help='What classification task is this '
                                                                                    'sample for?')
parser.add_argument('--filename', default=filename, type=str, help='What would you like the task file to be called?')

parsed_args = parser.parse_args()

json_tasks_path = os.path.join(LS_TASKS_PATH, parsed_args.filename)
LOGGING_FILENAME = os.path.join(LS_TASKS_PATH, '{}_log.txt'.format(parsed_args.filename))
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=LOGGING_FILENAME, filemode='w')


def main():
    random_query = """
        SELECT {columns} FROM {table} 
        ORDER BY RANDOM() 
        LIMIT {n};""".format(columns=', '.join(['id'] + parsed_args.columns), table=parsed_args.table, n=parsed_args.n)
    tasks_df = pd.read_sql(random_query, DATABASE_IP)

    logging.info("Successfully sampled {n} data points from {table}".format(n=parsed_args.n, table=parsed_args.table))
    tasks_df['table'] = parsed_args.table

    task_json = []
    for row in tasks_df.itertuples(index=False):
        data_dict = {"ref_id": row.id, "meta_info": {"table": row.table, "task": parsed_args.task}}
        data_dict.update(dict(zip(parsed_args.columns, [getattr(row, c) for c in parsed_args.columns])))
        task_json.append(
            {
                "id": row.id,
                "data": data_dict
            }
        )

    with open(json_tasks_path, 'w') as outfile:
        json.dump(task_json, outfile)

    logging.info("Saved {task} tasks in {path}".format(task=parsed_args.task, path=json_tasks_path))


if __name__ == '__main__':
    main()
