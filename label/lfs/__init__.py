from enum import Enum
import os


KEYWORDS_YAML_FILENAME = os.path.join('label', 'lfs', 'keywords.yaml')
ABSTAIN = -1


class FrameLabel(Enum):
    ST = 0
    Ec = 1
    En = 2
    ME = 3
    H = 4
    C = 5


class PipelineLabel(Enum):
    pass
