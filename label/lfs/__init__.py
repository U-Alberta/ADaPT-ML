from enum import Enum

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
