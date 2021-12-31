from enum import Enum


class CreatorApplyType(Enum):
    BeforeTrainBeforeLabel = 3
    BeforeTrainAfterLabel = 4
    OnceOnAllDataBeforeLabel = 5
    OnceOnAllDataAfterLabel = 6


class CreatorApplyTypeLabel(Enum):
    OnceOnAllData = 1
    BeforeTrain = 2
