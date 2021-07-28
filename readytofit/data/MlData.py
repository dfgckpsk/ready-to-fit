import pandas as pd
import numpy as np
from typing import List
from readytofit.tools.logging import logged


@logged
class MlData:

    def __init__(self,
                 feature_names: List[str],
                 features: np.array,
                 labels: np.array,
                 weights: pd.Series = None,
                 label_names=None,
                 indexes: np.array = None):
        self.feature_names = feature_names
        self.features = features
        self.labels = labels
        self.weights = weights
        self.label_names = label_names
        self.indexes = indexes
        self.index_to_int = {}
        for i, ind in enumerate(self.indexes):
            self.index_to_int[ind] = i

    def __len__(self):
        return self.features.shape[0]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.indexes = self.indexes.copy()
        result.index_to_int = dict(self.index_to_int)
        result.feature_names = self.feature_names.copy()
        result.label_names = self.label_names.copy() if self.label_names is not None else None
        result.features = self.features.copy()
        result.labels = self.labels.copy() if self.labels is not None else None
        result.weights = self.weights.copy() if self.weights is not None else None
        return result

    def _feature_index(self, feature_name):
        if type(feature_name) == list:
            indexes = []
            for f in feature_name:
                indexes.append(self.feature_names.index(f))
            return indexes
        return self.feature_names.index(feature_name)

    def remove_feature(self, feature: str):
        if feature in self.feature_names:
            ind = self._feature_index(feature)
            self.feature_names.remove(feature)
            self.features = np.delete(self.features, ind, 1)

    def validate_data(self):

        if self.labels is not None and self.features.shape[0] != self.labels.shape[0]:
            self._critical(f'features shape is {self.features.shape[0]}, '
                           f'but labels shape is {self.labels.shape[0]}')

        if self.weights is not None and self.features.shape[0] != self.weights.shape[0]:
            self._critical(f'features shape is {self.features.shape[0]}, '
                           f'but weights shape is {self.weights.shape[0]}')

    def get_features(self, indexes=None, feature=None):
        if feature is not None and feature not in self.feature_names and type(feature) == str or \
            type(feature) == list and not set(feature).issubset(set(self.feature_names)):
            self._critical(f'feature {feature} is not in features. Features - "{self.feature_names}"')

        self.validate_data()
        features = self.features
        if indexes is not None:
            indexes_ = list(map(lambda x: self.index_to_int[x], indexes))
            features = features[indexes_]
        if feature is not None:
            feature_i = self._feature_index(feature)
            features = features[:, feature_i]
        return np.copy(features)

    def get_targets(self, indexes=None):
        self.validate_data()
        if self.labels is None:
            return None
        if indexes is None:
            return np.copy(self.labels)
        indexes_ = list(map(lambda x: self.index_to_int[x], indexes))
        return np.copy(self.labels[indexes_])

    def get_indexes(self):
        return self.indexes

    def get_weights(self, indexes=None):
        if self.weights is None:
            return None
        if indexes is None:
            return np.copy(self.weights)
        indexes_ = list(map(lambda x: self.index_to_int[x], indexes))
        return np.copy(self.weights[indexes_])

    def update_labels(self, values):

        if self.features.shape[0] != len(values):
            self._critical('Features len is not equal to labels len')
            return

        self.labels = np.array(values)

    def update_weights(self, values):

        if self.features.shape[0] != len(values):
            self._critical('Features len is not equal to weights len')
            return

        self.weights = pd.Series(values, index=self.get_indexes())

    def update_features(self, values, name):

        if self.features.shape[0] != len(values):
            self._critical('Features len is not equal to values len')
            return

        if name in self.feature_names:
            column_ind = self._feature_index(name)
            self.features[:, column_ind] = values
        else:
            self.feature_names.append(name)
            self.features = np.column_stack((self.features,
                                             np.array(values)))

        if name not in self.feature_names:
            self.feature_names.append(name)
