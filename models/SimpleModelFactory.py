from .LGBMModel import LGMBModel
from .SLModel import SLModel
from tools.logging import logged


@logged
class SimpleModelFactory:

    """Factory for not ensembles"""

    def get_model(self, model_name: str, parameters: dict):

        if model_name == 'lgbm':
            return LGMBModel(parameters=parameters)
        elif model_name == 'slmodel':
            return SLModel(parameters=parameters)
        else:
            self._critical(f'Model with name {model_name} is not supported')
