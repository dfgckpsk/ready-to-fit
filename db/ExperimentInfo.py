from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExperimentInfo:
    exp_id: str
    uniq_count_runs: int
    count_runs: int
    min_date: datetime = None
    max_date: datetime = None
    max_split: int = None

    def to_dict(self):
        return self.__dict__
