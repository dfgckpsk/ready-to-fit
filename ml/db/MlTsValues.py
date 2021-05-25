from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MlTsValues:
    run_id: datetime
    value_name: str
    split_num: int = None
    indexes: list = field(default_factory=list)
    float_values: list = field(default_factory=list)
    int_values: list = field(default_factory=list)

    def to_dict(self):
        return self.__dict__
