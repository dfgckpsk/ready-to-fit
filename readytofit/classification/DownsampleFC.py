from readytofit.data.FeatureCreator import FeatureCreator, MlData
from readytofit.data.MlDataFactory import MlDataFactory
import numpy as np


class DownsampleFC(FeatureCreator):

    def apply(self, ml_data: MlData):

        labels = ml_data.get_targets()
        indexes = ml_data.get_indexes()
        unique_classes, counts = np.unique(labels, return_counts=True)
        min_elements_count = min(counts)

        new_features = []
        new_labels = []
        new_indexes = []
        for class_id in unique_classes:
            class_indexes = indexes[np.where(labels == class_id)[0]]
            randomed_indexes = np.random.choice(class_indexes, min_elements_count, replace=False)

            class_features = ml_data.get_features(indexes=randomed_indexes)
            class_labels = ml_data.get_targets(indexes=randomed_indexes)

            new_features.append(class_features)
            new_labels.append(class_labels)
            new_indexes.append(randomed_indexes)

        new_indexes = np.concatenate(new_indexes)
        new_features = np.concatenate(new_features)
        new_labels = np.concatenate(new_labels)

        ml_data_ = MlDataFactory().ml_data_from_numpy(new_features, new_labels, indexes=new_indexes, label_names=ml_data.label_names)

        return ml_data_
