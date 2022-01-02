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
                 categorical_enum: Enum.__class__ = None,
                 default_value = None):
        self.name = name
        self.type = _type
        self.boundaries = boundaries
        self.nullable = nullable
        self.categorical_enum = categorical_enum

        self.default_value = default_value
        if self.default_value is None:
            if self.type == ParameterTypes.Bool:
                self.default_value = False
            elif self.type in (ParameterTypes.Float, ParameterTypes.Integer) and self.boundaries is not None:
                self.default_value = self.boundaries[0]
            elif self.type == ParameterTypes.Categorical:
                self.default_value = [e.value for e in categorical_enum][0]

        # if _type == ParameterTypes.Categorical
