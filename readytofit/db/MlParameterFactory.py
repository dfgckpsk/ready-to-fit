from ..db.MlParameter import MlParameter, MlParameterType
from datetime import datetime
from ..tools.types import isdigit


class MLParameterFactory:

    @staticmethod
    def from_dict(parameters: dict, run_id: datetime, parameter_type: MlParameterType, split_num: int = None):
        ml_parameters = []
        for key in parameters:
            value = parameters.get(key)
            float_value = value if isdigit(value) else None
            str_value = value if not isdigit(value) else None
            ml_parameter = MlParameter(run_id=run_id,
                                       parameter_type=parameter_type,
                                       parameter_name=key,
                                       split_num=split_num,
                                       float_value=float_value,
                                       str_value=str_value)
            ml_parameters.append(ml_parameter)
        return ml_parameters
