import os
import logging

DATABASE_IP = os.environ['DATABASE_IP']
LABEL_STUDIO_DIRECTORY = '/label_studio'
PV_TASKS_DIRECTORY = os.path.join(LABEL_STUDIO_DIRECTORY, 'example_tasks')
LOGGING_FILENAME = os.path.join(LABEL_STUDIO_DIRECTORY, 'dev_log.txt')

EXAMPLE_VOCAB = [
    "cat",
    "dog",
    "bird",
]

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=LOGGING_FILENAME, filemode='w')
