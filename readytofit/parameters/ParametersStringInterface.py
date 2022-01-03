from ..tools.logging import logged
from .BaseParameter import ParameterTypes, BaseParameter
import re
from typing import List


@logged
class ParametersStringInterface:

    def load_parameters(self, parameters_str: str, parameter_base_list: List[BaseParameter]):

        out_parameters = {}
        parameter_base_dict = dict(map(lambda x: (x.name, x), parameter_base_list))
        parameters = re.findall('(\w+).*: ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|True|False)', parameters_str, re.IGNORECASE)
        for parameter_name, parameter_value in parameters:
            parameter_base = parameter_base_dict.get(parameter_name)
            if parameter_base is None or parameter_value.lower() in ('none', 'null'):
                continue
            try:
                parameter_value_ = None
                boundaries = parameter_base.boundaries
                if parameter_base.type == ParameterTypes.Bool:
                    parameter_value_ = 'True' == parameter_value

                elif parameter_base.type == ParameterTypes.Float:
                    parameter_value_ = float(parameter_value)
                    if boundaries is not None:
                        parameter_value_ = max(min(parameter_value_, boundaries[1]), boundaries[0])

                elif parameter_base.type == ParameterTypes.Integer:
                    parameter_value_ = int(parameter_value)
                    if boundaries is not None:
                        parameter_value_ = max(min(parameter_value_, boundaries[1]), boundaries[0])

                elif parameter_base.type == ParameterTypes.Categorical:
                    parameter_value_ = int(parameter_value)
                    parameter_value_ = parameter_base.categorical_enum(parameter_value_)

                if parameter_value_ is not None:
                    out_parameters[parameter_name] = parameter_value_
            except BaseException as e:
                self._warning(f'{e}')

        return out_parameters

    def get_parameters_string(self, parameter_base_list: List[BaseParameter]):

        """
            Convert list of parameters to string for settings
        """

        result_string = ''
        for parameter_base in parameter_base_list:
            null_str = '(null)' if parameter_base.nullable else ''
            if parameter_base.type in (ParameterTypes.Integer, ParameterTypes.Float):
                result_string += f'{parameter_base.name}' \
                                 f'{parameter_base.boundaries}' \
                                 f'{null_str}: ' \
                                 f'{parameter_base.default_value}'

            elif parameter_base.type == ParameterTypes.Categorical:
                categorical_values = [f'{e.name}({e.value})' for e in parameter_base.categorical_enum]
                categorical_values_str = ','.join(categorical_values)
                result_string += f'{parameter_base.name}' \
                                 f'({categorical_values_str})' \
                                 f'{null_str}: ' \
                                 f'{parameter_base.default_value}'

            elif parameter_base.type == ParameterTypes.Bool:
                result_string += f'{parameter_base.name}' \
                                 f'{null_str}: ' \
                                 f'{parameter_base.default_value}'

            result_string += '\n'
        return result_string
