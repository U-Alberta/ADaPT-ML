import argparse
from ls import CRATE_DB_IP
import pandas as pd
import json
from datetime import datetime
import os

parser = argparse.ArgumentParser(description='Sample a number of tweets from a table to annotate.')
parser.add_argument('table', type=str, help='Table name that stores the tweets.')
parser.add_argument('n', type=int, help='Number of tweets to sample.')
parser.add_argument('task', type=str, help='What task is this sample for? {pv}')
parser.add_argument('--filename', default=str(datetime.now()), type=str, help='What would you like the task file to be called?')

parsed_args = parser.parse_args()

if parsed_args.task == 'pv':
    from ls import PV_TASKS_DIRECTORY as TASK_DIRECTORY


def main():
    random_query = """
    SELECT id, tweet FROM {table} 
    ORDER BY RANDOM() 
    LIMIT {n};""".format(table=parsed_args.table, n=parsed_args.n)
    df = pd.read_sql(random_query, CRATE_DB_IP)

    df['table'] = parsed_args.table
    task_list = []

    for row in df.iterrows():
        task = {
            "tweet_text": row['tweet'],
            "ref_id": row['id'],
            "meta_info": {
                "table": row['table']
            }
        }
        task_list.append(task)

    with open(os.path.join(TASK_DIRECTORY, parsed_args.filename), 'w') as outfile:
        json.dump(task_list, outfile)


if __name__ == '__main__':
    main()
