from ..data.FeatureCreator import FeatureCreator, MlData
from ..data.MlDataFactory import MlDataFactory
import numpy as np
import pandas as pd


class ByColumnsNormalizationFC(FeatureCreator):

    def apply(self, ml_data: MlData):

        new_feature_names = []
        new_features = []
        left_columns = self.left_columns(ml_data.feature_names)

        for feature in self.apply_feature:
            selected_features = list(filter(lambda x: feature in x, ml_data.feature_names))
            f = ml_data.get_features(feature=selected_features)
            std = np.std(f.astype(float), axis=1).reshape(-1, 1)
            std[std == 0] = 1
            mean = np.mean(f, axis=1).reshape(-1, 1)
            new_features.append((f - mean) / std)
            new_feature_names += selected_features
        new_features = np.concatenate(new_features, axis=1)

        features = pd.DataFrame(new_features, columns=new_feature_names, index=ml_data.get_indexes())
        for column in left_columns:
            features[column] = ml_data.get_features(feature=column)
            new_feature_names.append(column)
        ml_data_ = MlDataFactory().ml_data_from_numpy(features,
                                                      ml_data.get_targets(),
                                                      feature_columns=new_feature_names,
                                                      indexes=ml_data.get_indexes(),
                                                      weights=ml_data.get_weights(),
                                                      label_names=ml_data.label_names)
        return ml_data_

    def left_columns(self, current_columns):

        left_columns = []
        for current_column in current_columns:
            if not any(map(lambda x: x in current_column, self.apply_feature)) and current_column not in left_columns:
                left_columns.append(current_column)
        return left_columns
