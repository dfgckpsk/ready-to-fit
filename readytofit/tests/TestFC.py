from unittest import TestCase
from enum import Enum
import numpy as np
import pandas as pd
from ..preprocessing.RenameLabelsFC import RenameLabelsFC
from ..preprocessing.LeaveLabelsFC import LeaveLabelsFC
from ..preprocessing.SlideFC import SlideFC
from ..preprocessing.ByColumnsNormalizationFC import ByColumnsNormalizationFC
from ..preprocessing.SimpleSampleWeightsFC import SimpleSampleWeightsFC
from ..data.MlDataFactory import MlDataFactory
from ..data.FeatureCreator import CreatorApplyType, FeatureCreator, List
from ..parameters.BaseParameter import ParameterTypes, BaseParameter
from ..parameters.ParametersStringInterface import ParametersStringInterface
import random


class TestFC(TestCase):

    def generate_ml_data(self, row_count):
        features = []
        for i in range(row_count):
            features.append([round(random.random(), 3), round(random.random(), 3)])
        features = np.array(features)
        ml_data = MlDataFactory().ml_data_from_numpy(features,
                                                           target=np.array([0] * row_count),
                                                           indexes=list(range(101, 101+row_count)))
        return ml_data

    def test_slide_fc(self):
        row_count = 100000
        ml_data = self.generate_ml_data(row_count)

        slide_fc = SlideFC(parameters=dict(slide_values=2))
        ml_data_ = slide_fc.apply(ml_data)
        assert len(ml_data_.feature_names) == 6
        assert len(ml_data) - 2 == len(ml_data_)
        assert ml_data_.get_features(indexes=[ml_data_.get_indexes()[2]], feature='feature_0_2') == \
               ml_data_.get_features(indexes=[ml_data_.get_indexes()[0]], feature='feature_0')

        slide_fc = SlideFC(parameters=dict(slide_values=2), apply_feature=['feature_0'])
        ml_data_ = slide_fc.apply(ml_data)
        assert len(ml_data_.feature_names) == 4
        assert len(ml_data) - 2 == len(ml_data_)
        assert ml_data_.get_features(indexes=[ml_data_.get_indexes()[2]], feature='feature_0_2') == \
               ml_data_.get_features(indexes=[ml_data_.get_indexes()[0]], feature='feature_0')

    def test_normalization_sc(self):
        row_count = 100000
        ml_data = self.generate_ml_data(row_count)

        slide_fc = SlideFC(parameters=dict(slide_values=2))
        by_columns_normalization_fc = ByColumnsNormalizationFC(apply_feature=['feature_0', 'feature_1'],
                                                               creator_apply_type=CreatorApplyType.BeforeTrainBeforeLabel)

        ml_data_ = slide_fc.apply(ml_data)
        wide_features = pd.DataFrame(ml_data_.get_features(), columns=ml_data_.feature_names)
        ml_data_ = by_columns_normalization_fc.apply(ml_data_)

        feature_0_mean = wide_features[['feature_0', 'feature_0_1', 'feature_0_2']].mean(axis=1)
        feature_0_std = wide_features[['feature_0', 'feature_0_1', 'feature_0_2']].std(axis=1, ddof=0)
        feature_1_mean = wide_features[['feature_1', 'feature_1_1', 'feature_1_2']].mean(axis=1)
        feature_1_std = wide_features[['feature_1', 'feature_1_1', 'feature_1_2']].std(axis=1, ddof=0)

        feature_0 = (wide_features['feature_0'] - feature_0_mean) / feature_0_std
        feature_0_1 = (wide_features['feature_0_1'] - feature_0_mean) / feature_0_std
        feature_1 = (wide_features['feature_1'] - feature_1_mean) / feature_1_std
        assert (feature_0.values == ml_data_.get_features(feature='feature_0')).all()
        assert (feature_0_1.values == ml_data_.get_features(feature='feature_0_1')).all()
        assert (feature_1.values == ml_data_.get_features(feature='feature_1')).all()

    def test_weights(self):
        ml_data = self.generate_ml_data(10)
        ml_data.update_labels([1, 0, 0, 1, 1, 0, 0, 0, 0, 0])

        sample_weights_fc = SimpleSampleWeightsFC(apply_feature='', creator_apply_type=CreatorApplyType.BeforeTrainBeforeLabel)
        ml_data = sample_weights_fc.apply(ml_data)
        assert ([0.7, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3] == ml_data.get_weights()).all()

        ml_data.update_labels([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        sample_weights_fc = SimpleSampleWeightsFC(apply_feature='', creator_apply_type=CreatorApplyType.BeforeTrainBeforeLabel)
        ml_data = sample_weights_fc.apply(ml_data)
        assert ([0.8, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2] == ml_data.get_weights()).all()

    def test_leave_label_fc(self):
        ml_data = self.generate_ml_data(10)
        ml_data.update_labels([1, 0, 0, 1, 1, 0, 0, 2, 0, 0])

        leaved_ml_data = LeaveLabelsFC(leave_labels=[1,2]).apply(ml_data)

        assert len(leaved_ml_data) == 4
        assert (leaved_ml_data.get_features() == ml_data.get_features([101, 104, 105, 108])).all()

        leaved_ml_data = LeaveLabelsFC(leave_labels=[1]).apply(ml_data)

        assert len(leaved_ml_data) == 3
        assert (leaved_ml_data.get_features() == ml_data.get_features([101, 104, 105])).all()

    def test_rename_label_fc(self):
        ml_data = self.generate_ml_data(10)
        ml_data.update_labels([1, 0, 0, 1, 1, 0, 0, 2, 0, 0])

        renamed_ml_data = RenameLabelsFC(new_labels={0: 12}).apply(ml_data)
        assert ([1, 0, 0, 1, 1, 0, 0, 2, 0, 0] == renamed_ml_data.get_targets()).all()

        renamed_ml_data = RenameLabelsFC(new_labels={0: 10, 1: 11, 2: 12}).apply(ml_data)
        assert ([11, 10, 10, 11, 11, 10, 10, 12, 10, 10] == renamed_ml_data.get_targets()).all()

    def test_parameters(self):

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

        feature = TestParameterFc([], CreatorApplyType.BeforeTrainBeforeLabel)
        param_str = ParametersStringInterface().get_parameters_string(feature.get_parameters_description())
        print(param_str)

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
                        int_param_2(0, 10)(null): 7
                        int_param_3(0, 10): 12
                        cat_param_1(p1(1),p2(2),p3(3)): 1
                        cat_param_2(p1(1),p2(2),p3(3))(null): 1
                        bool_param_1: True
                        bool_param_2: False
                        bool_param_3(null): False"""
        test_fc = TestParameterFc([], CreatorApplyType.BeforeTrainBeforeLabel)
        ParametersStringInterface().load_parameters(result_str,
                                                    feature.get_parameters_description())
