import itertools


class GridSearch:

    """
    Grid parameters generator
    """

    def __init__(self, param_grid, **kwargs):
        self.param_grid = param_grid
        self.length = None

    def __len__(self):

        if self.length is None:
            _, combinations = self.get_combinations()
            self.length = len(list(combinations))
        return self.length

    def get_combinations(self):

        param_names = sorted(self.param_grid)

        combinations = itertools.product(*(self.param_grid[name] for name in param_names))

        return param_names, combinations

    def next_param(self):

        """
        Generate all grid parameters combinations
        :return:
        """

        param_names, combinations = self.get_combinations()

        for combination in combinations:

            out_dict = {}

            for param, value in zip(param_names, combination):

                out_dict[param] = value

            yield out_dict
