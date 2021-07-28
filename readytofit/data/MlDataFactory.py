from readytofit.data.MlData import MlData
import numpy as np


class MlDataFactory:

    def ml_data_from_numpy(self, features, target, feature_columns=None, indexes=None, weights=None, label_names=None):
        if feature_columns is None:
            feature_columns = list(map(lambda x: f'feature_{x}', range(features.shape[1])))
        features = np.array(features)
        indexes = np.array(list(range(len(features)))) if indexes is None else np.array(indexes)
        target = np.array(target) if target is not None else None
        return MlData(list(feature_columns), features, target, weights=weights, label_names=label_names, indexes=indexes)

    def ml_data_from_ml_data(self, ml_data: MlData, indexes):
        return self.ml_data_from_numpy(ml_data.get_features(indexes),
                                       ml_data.get_targets(indexes),
                                       ml_data.feature_names,
                                       indexes,
                                       weights=ml_data.get_weights(indexes),
                                       label_names=ml_data.label_names)
