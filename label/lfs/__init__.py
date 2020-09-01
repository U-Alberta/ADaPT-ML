from enum import Enum
import os

LEXICONS_PATH = os.path.join('label', 'resources', 'lexicons.db')
KEYWORDS_YAML_FILENAME = os.path.join('label', 'lfs', 'keywords.yaml')
ABSTAIN = -1


class ValueLabel(Enum):
    SE = 0      # Security
    CO = 1      # Conformity
    TR = 2      # Tradition
    BE = 3      # Benevolence
    UN = 4      # Universalism
    SD = 5      # Self-direction
    ST = 6      # Stimulation
    HE = 7      # Hedonism
    AC = 8      # Achievement
    PO = 9      # Power


class FrameLabel(Enum):
    ST = 0
    Ec = 1
    En = 2
    ME = 3
    H = 4
    C = 5


class ATSCLabels(Enum):
    NEG = 0
    NEU = 1
    POS = 2


class ArousalLabel(Enum):
    LOW = 0
    MED = 1
    HIGH = 2


class DominanceLabel(Enum):
    LOW = 0
    MED = 1
    HIGH = 2
