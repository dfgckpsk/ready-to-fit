import random
from .GridSearch import GridSearch
from readytofit.tools.types import isfloat, isint


class ParamType:
    STR = 1
    FLOAT = 2
    INT = 3


class RandomSearch(GridSearch):

    """
    Grid parameters generator
    """

    def __init__(self, param_grid, **kwargs):
        GridSearch.__init__(self, param_grid, **kwargs)
        self.combinations_per_parameter = kwargs.get('combinations_per_parameter', 10)

        self.just_digits = {}
        param_names = sorted(self.param_grid)
        for param in param_names:
            type = ParamType.STR
            if all(map(lambda x: isfloat(x), self.param_grid[param])):
                type = ParamType.FLOAT
            elif all(map(lambda x: isint(x), self.param_grid[param])):
                type = ParamType.INT
            self.just_digits[param] = type

    def get_combinations(self):

        param_names = sorted(self.param_grid)

        parameters_count = len(list(filter(lambda x: len(self.param_grid[x]) > 1, param_names)))
        combinations = []
        for _ in range(self.combinations_per_parameter * parameters_count):
            combination = []
            for param in param_names:
                if self.just_digits[param] != ParamType.STR:
                    digits = list(map(lambda x: float(x), self.param_grid[param]))
                    min_ = min(digits)
                    max_ = max(digits)
                    value = random.random() * (max_ - min_) + min_
                    if self.just_digits[param] == ParamType.INT:
                        value = int(value)
                else:
                    value = random.choice(self.param_grid[param])
                combination.append(value)
            combinations.append(combination)

        return param_names, combinations
