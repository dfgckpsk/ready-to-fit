from ..data.FeatureCreator import FeatureCreator, MlData
from ..data.MlDataFactory import MlDataFactory
from ..parameters.BaseParameter import BaseParameter, ParameterTypes
import pandas as pd


class SlideFC(FeatureCreator):

    def __init__(self, apply_feature=None, to_column=None, parameters={}):
        FeatureCreator.__init__(self, apply_feature, to_column, parameters=parameters)
        self.slide_values = self.parameters.get('slide_values')

    def __parameters_description(self):
        return [BaseParameter('slide_values', ParameterTypes.Integer, (0, 100))]

    def apply(self, ml_data: MlData):

        apply_feature = ml_data.feature_names if self.apply_feature is None else self.apply_feature
        features = pd.DataFrame(ml_data.get_features(),
                                index=ml_data.get_indexes(),
                                columns=ml_data.feature_names)
        for feature_name in apply_feature:
            for shift in range(1, self.slide_values + 1):
                features[f'{feature_name}_{shift}'] = features[feature_name].shift(shift)

        features = features.iloc[self.slide_values:]
        labels = ml_data.get_targets(features.index)
        ml_data_ = MlDataFactory().ml_data_from_numpy(features.values, labels, features.columns, features.index,
                                                      weights=ml_data.get_weights(features.index),
                                                      label_names=ml_data.label_names)
        return ml_data_
