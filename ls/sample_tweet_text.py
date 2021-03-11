import argparse
from ls import CRATE_DB_IP
import pandas as pd
import json
from datetime import datetime
import os
import logging

date_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = date_str+'.json'

parser = argparse.ArgumentParser(description='Sample a number of tweets from a table to annotate.')
parser.add_argument('table', type=str, help='Table name that stores the tweets.')
parser.add_argument('n', type=int, help='Number of tweets to sample.')
parser.add_argument('task', type=str, help='What task is this sample for? {pv}')
parser.add_argument('--filename', default=filename, type=str, help='What would you like the task file to be called?')

parsed_args = parser.parse_args()

if parsed_args.task == 'pv':
    from ls import PV_TASKS_DIRECTORY as TASK_DIRECTORY

task_path = os.path.join(TASK_DIRECTORY, parsed_args.filename)


def main():
    random_query = """
        SELECT id, tweet FROM {table} 
        ORDER BY RANDOM() 
        LIMIT {n};""".format(table=parsed_args.table, n=parsed_args.n)
    df = pd.read_sql(random_query, CRATE_DB_IP)

    logging.info("Successfully sampled {n} tweets from {table}".format(n=parsed_args.n, table=parsed_args.table))
    df['table'] = parsed_args.table
    task_list = []

    for row in df.iterrows():
        task = {
            "tweet_text": row[1]['tweet'],
            "ref_id": row[1]['id'],
            "meta_info": {
                "table": row[1]['table']
            }
        }
        task_list.append(task)

    with open(task_path, 'w') as outfile:
        json.dump(task_list, outfile)

    logging.info("Saved {task} tasks in {path}".format(task=parsed_args.task, path=task_path))


if __name__ == '__main__':
    main()
