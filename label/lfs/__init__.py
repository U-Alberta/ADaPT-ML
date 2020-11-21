from enum import Enum

ABSTAIN = -1


class ValueLabel(Enum):
    SE = 0      # Security
    security = 0
    CO = 1      # Conformity
    conformity = 1
    TR = 2      # Tradition
    tradition = 2
    BE = 3      # Benevolence
    benevolence = 3
    UN = 4      # Universalism
    universalism = 4
    SD = 5      # Self-direction
    self_direction = 5
    ST = 6      # Stimulation
    stimulation = 6
    HE = 7      # Hedonism
    hedonism = 7
    AC = 8      # Achievement
    achievement = 8
    PO = 9      # Power
    power = 9
