import argparse
import os
import logging

parser = argparse.ArgumentParser(description='Train a classifier.')
parser.add_argument('--model', default=0, help='train an mlogit model (0)')
parser.add_argument('--eval', default=False, help='evaluate the model')
parsed_args = parser.parse_args()

BUNDLED_FILENAME = os.path.join('classify', 'resources', 'bundled.pkl')

DEMO_BUNDLED_FILENAME = os.path.join('demo', 'demo_bundled.pkl')

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
