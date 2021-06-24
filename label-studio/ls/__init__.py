import os
import logging

CRATE_DB_IP = os.environ['CRATE_DB_IP']
LABEL_STUDIO_DIRECTORY = '/label_studio'
PV_TASKS_DIRECTORY = os.path.join(LABEL_STUDIO_DIRECTORY, 'pv_tasks')
LOGGING_FILENAME = os.path.join(LABEL_STUDIO_DIRECTORY, 'dev_log.txt')

PV_VOCAB = [
    "security",
    "conformity",
    "tradition",
    "benevolence",
    "universalism",
    "self_direction",
    "stimulation",
    "hedonism",
    "achievement",
    "power"
]

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=LOGGING_FILENAME, filemode='w')
