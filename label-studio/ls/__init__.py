import os

DATABASE_IP = os.environ['DATABASE_IP']
LS_TASKS_PATH = '/tasks'
LS_ANNOTATIONS_PATH = '/annotations'

LABEL_PREFIX = 'worker_'
CLASSIFICATION_TASKS = {
    'example': ('cat', 'dog', 'bird', 'horse', 'snake')
}

for task in CLASSIFICATION_TASKS:
    try:
        os.mkdir(os.path.join(LS_ANNOTATIONS_PATH, task))
    except FileExistsError:
        pass
