from enum import Enum


class CreatorApplyType(Enum):
    OnceOnAllData = 1
    BeforeTrain = 2
    BeforeTrainBeforeLabel = 3
    BeforeTrainAfterLabel = 4
    OnceOnAllDataBeforeLabel = 5
    OnceOnAllDataAfterLabel = 6
