import os
import logging

CRATE_DB_IP = os.environ['CRATE_DB_IP']
PV_TASKS_DIRECTORY = '/pv_tasts'
LOGGING_FILENAME = '/dev_log.txt'
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename=LOGGING_FILENAME, filemode='w')
