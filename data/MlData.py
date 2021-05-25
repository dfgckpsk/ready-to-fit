import pandas as pd
from typing import List
from tools.logging import logged


@logged
class MlData:

    def __init__(self, feature_names: List[str], features: pd.DataFrame, labels: pd.Series, weights: pd.Series = None, label_names=None):
        self.feature_names = feature_names
        self.features = features
        self.labels = labels
        self.weights = weights
        self.label_names = label_names

    def __len__(self):
        return self.features.shape[0]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.feature_names = self.feature_names.copy()
        result.label_names = self.label_names.copy() if self.label_names is not None else None
        result.features = self.features.copy()
        result.labels = self.labels.copy() if self.labels is not None else None
        result.weights = self.weights.copy() if self.weights is not None else None
        return result

    def remove_feature(self, feature: str):
        if feature in self.features.columns:
            del self.features[feature]
        if feature in self.feature_names:
            self.feature_names.remove(feature)

    def validate_data(self):

        if self.labels is not None and self.features.shape[0] != self.labels.shape[0]:
            self._critical(f'features shape is {self.features.shape[0]}, '
                        f'but labels shape is {self.labels.shape[0]}')

        if self.weights is not None and self.features.shape[0] != self.weights.shape[0]:
            self._critical(f'features shape is {self.features.shape[0]}, '
                        f'but weights shape is {self.weights.shape[0]}')

        for feature_name in self.feature_names:
            if feature_name not in self.features.columns:
                self._critical(f'feature {feature_name} is not in features')

    def get_features(self, indexes=None, feature=None):
        if feature is not None and feature not in self.feature_names and type(feature) == str or \
            type(feature) == list and not set(feature).issubset(set(self.feature_names)):
            self._critical(f'feature {feature} is not in features. Features - "{self.feature_names}"')

        self.validate_data()
        features = self.features
        if indexes is not None:
            features = features.loc[indexes]
        if feature is not None:
            features = features[feature]
        return features.copy().values

    def get_targets(self, indexes=None):
        self.validate_data()
        if self.labels is None:
            return None
        if indexes is None:
            return self.labels.copy().values
        return self.labels.loc[indexes].copy().values

    def get_indexes(self):
        return self.features.index

    def get_weights(self, indexes=None):
        if self.weights is None:
            return None
        if indexes is None:
            return self.weights.copy().values
        return self.weights.loc[indexes].copy().values

    def update_labels(self, values):

        if self.features.shape[0] != len(values):
            self._critical('Features len is not equal to labels len')
            return

        self.labels = pd.Series(values, index=self.get_indexes())

    def update_weights(self, values):

        if self.features.shape[0] != len(values):
            self._critical('Features len is not equal to weights len')
            return

        self.weights = pd.Series(values, index=self.get_indexes())

    def update_features(self, values, name):

        if self.features.shape[0] != len(values):
            self._critical('Features len is not equal to values len')
            return

        self.features[name] = values

        if name not in self.feature_names:
            self.feature_names.append(name)
