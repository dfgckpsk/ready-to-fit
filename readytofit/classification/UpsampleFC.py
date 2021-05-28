from readytofit.data.FeatureCreator import FeatureCreator, MlData
from readytofit.data.MlDataFactory import MlDataFactory
import numpy as np


class UpsampleFC(FeatureCreator):

    def __init__(self, apply_feature=None, to_column=None, apply_after_label=False, upsample_coef=1.0):
        FeatureCreator.__init__(self, apply_feature, to_column, apply_after_label)
        self.upsample_coef = upsample_coef

    def apply(self, ml_data: MlData):

        labels = ml_data.get_targets()
        indexes = ml_data.get_indexes()
        unique_classes, counts = np.unique(labels, return_counts=True)
        max_elements_count = max(counts)
        max_class = unique_classes[np.argmax(counts)]

        new_features = []
        new_labels = []
        new_indexes = []
        for class_id in unique_classes:
            class_indexes = indexes[np.where(labels == class_id)[0]]
            random_count = int(max_elements_count * self.upsample_coef) if class_id != max_class else max_elements_count
            randomed_indexes = np.random.choice(class_indexes, random_count, replace=True)

            class_features = ml_data.get_features(indexes=randomed_indexes)
            class_labels = ml_data.get_targets(indexes=randomed_indexes)

            new_features.append(class_features)
            new_labels.append(class_labels)
            new_indexes.append(randomed_indexes)

        new_indexes = np.concatenate(new_indexes)
        new_features = np.concatenate(new_features)
        new_labels = np.concatenate(new_labels)

        ml_data_ = MlDataFactory().ml_data_from_numpy(new_features, new_labels, indexes=new_indexes)

        return ml_data_
