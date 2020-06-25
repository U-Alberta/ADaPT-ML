from enum import Enum
import os


KEYWORDS_YAML_FILENAME = os.path.join('label', 'lfs', 'keywords.yaml')


class Label(Enum):
    ABSTAIN = -1
    ST_P = 0
    ST_N = 1
    Ec_P = 2
    Ec_N = 3
    En_P = 4
    En_N = 5
    ME_P = 6
    ME_N = 7
    H_P = 8
    H_N = 9
    C_P = 10
    C_N = 11
