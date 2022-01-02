from enum import Enum
from ..data.FeatureCreator import CreatorApplyType, FeatureCreator, BaseParameter, List
from ..parameters.BaseParameter import ParameterTypes
from ..parameters.ParametersStringInterface import ParametersStringInterface


def test_parameters():

    class CatParam(Enum):
        p1 = 1
        p2 = 2
        p3 = 3

    class TestParameterFc(FeatureCreator):
        def _parameters_description(self) -> List[BaseParameter]:
            return [BaseParameter('int_param_1', ParameterTypes.Integer, boundaries=(0, 10)),
                    BaseParameter('int_param_2', ParameterTypes.Integer, boundaries=(0, 10), nullable=True),
                    BaseParameter('int_param_3', ParameterTypes.Integer, boundaries=(0, 10), default_value=12),

                    BaseParameter('cat_param_1', ParameterTypes.Categorical, categorical_enum=CatParam),
                    BaseParameter('cat_param_2', ParameterTypes.Categorical, categorical_enum=CatParam, nullable=True),

                    BaseParameter('bool_param_1', ParameterTypes.Bool, default_value=True),
                    BaseParameter('bool_param_2', ParameterTypes.Bool, default_value=False),
                    BaseParameter('bool_param_3', ParameterTypes.Bool, default_value=False, nullable=True)
                    ]

    parameters = TestParameterFc([], CreatorApplyType.BeforeTrainBeforeLabel).get_parameters_description()
    param_str = ParametersStringInterface().get_parameters_string(parameters)

    param_str_list = param_str.split('\n')
    assert param_str_list[0] == 'int_param_1(0, 10): 0'
    assert param_str_list[1] == 'int_param_2(0, 10)(null): 0'
    assert param_str_list[2] == 'int_param_3(0, 10): 12'
    assert param_str_list[3] == 'cat_param_1(p1(1),p2(2),p3(3)): 1'
    assert param_str_list[4] == 'cat_param_2(p1(1),p2(2),p3(3))(null): 1'
    assert param_str_list[5] == 'bool_param_1: True'
    assert param_str_list[6] == 'bool_param_2: False'
    assert param_str_list[7] == 'bool_param_3(null): False'

    result_str = """int_param_1(0, 10): 6
                    int_param_2(0, 10)(null): -1
                    int_param_3(0, 10): 12
                    cat_param_1(p1(1),p2(2),p3(3)): 1
                    cat_param_2(p1(1),p2(2),p3(3))(null): 2
                    bool_param_1: True
                    bool_param_2: False"""
    test_fc = ParametersStringInterface()
    out_parameters = test_fc.load_parameters(result_str, parameters)
    assert len(out_parameters) == 7
    assert out_parameters['int_param_1'] == 6
    assert out_parameters['int_param_2'] == 0
    assert out_parameters['int_param_3'] == 10
    assert out_parameters['cat_param_1'] == CatParam.p1
    assert out_parameters['cat_param_2'] == CatParam.p2
    assert out_parameters['bool_param_1'] is True
    assert out_parameters['bool_param_2'] is False
