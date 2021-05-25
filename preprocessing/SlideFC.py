from data.FeatureCreator import FeatureCreator, MlData
from data.MlDataFactory import MlDataFactory
import pandas as pd


class SlideFC(FeatureCreator):

    def __init__(self, apply_feature=None, to_column=None, apply_after_label=False, slide_values=0):
        FeatureCreator.__init__(self, apply_feature, to_column, apply_after_label)
        self.slide_values = slide_values

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
