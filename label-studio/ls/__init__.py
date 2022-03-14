import os

DATABASE_IP = os.environ['DATABASE_IP']
LS_TASKS_PATH = '/tasks'
LS_ANNOTATIONS_PATH = '/annotations'

LABEL_PREFIX = 'worker_'
CLASSIFICATION_TASKS = {
    'example': ('cat', 'dog', 'bird', 'horse', 'snake'),
    'framing': ("settled_science", "uncertain_science", "political_or_ideological_struggle", "disaster", "opportunity",
                "economic", "morality_and_ethics", "role_of_science", "security", "health")
}

for task in CLASSIFICATION_TASKS:
    try:
        os.mkdir(os.path.join(LS_ANNOTATIONS_PATH, task))
    except FileExistsError:
        pass
