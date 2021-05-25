from .LGBMModel import LGMBModel
from .SLModel import SLModel
from models.StackModel import StackModel
from tools.logging import logged


@logged
class ModelFactory:

    def get_model(self, model_name: str, parameters: dict):

        if model_name == 'lgbm':
            return LGMBModel(parameters=parameters)
        elif model_name == 'slmodel':
            return SLModel(parameters=parameters)
        elif model_name == 'stack_model':
            return StackModel(parameters=parameters)
        else:
            self._critical(f'Model with name {model_name} is not supported')
