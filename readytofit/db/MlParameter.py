from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MlParameter:
    run_id: datetime
    parameter_name: str
    split_num: int = None
    str_value: str = None
    float_value: float = None

    def to_dict(self):
        return self.__dict__
