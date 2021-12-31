from ..tools.logging import logged
from enum import Enum


class ParameterTypes(Enum):

    Categorical = 1
    Integer = 2
    Float = 3
    Bool = 4


@logged
class BaseParameter:

    def __init__(self,
                 name: str,
                 _type: ParameterTypes,
                 boundaries: tuple = None,
                 nullable: bool = False,
                 categorical_enum = None):
        self.name = name
        self.type = _type
        self.boundaries = boundaries
        self.nullable = nullable
        self.categorical_enum = categorical_enum

        # if _type == ParameterTypes.Categorical
