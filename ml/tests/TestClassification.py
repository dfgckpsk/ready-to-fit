from unittest import TestCase
from ml.data.MlDataFactory import MlDataFactory
from ml.classification.DownsampleFC import DownsampleFC
from ml.classification.UpsampleFC import UpsampleFC
import numpy as np


class TestClassification(TestCase):

    def generate_imbalance_data(self, class_ratios, row_count):

        features = []
        labels = []
        for class_index, class_ratio in enumerate(class_ratios):
            features_class = [[f'f_{class_index}']] * int(row_count*class_ratio)
            labels_class = [class_index] * int(row_count*class_ratio)
            features += features_class
            labels += labels_class
        return np.array(features), np.array(labels)

    def test_downsample(self):

        row_count = 100000
        class_ratios = [0.7, 0.2, 0.1]
        features, labels = self.generate_imbalance_data(class_ratios, row_count)
        ml_data = MlDataFactory().ml_data_from_numpy(features, labels, indexes=list(range(101, 101+row_count)))

        ml_data_ = DownsampleFC().apply(ml_data)

        new_labels = ml_data_.get_targets()
        class_0_count = len(list(filter(lambda x: x == 0, new_labels)))
        class_1_count = len(list(filter(lambda x: x == 1, new_labels)))
        class_2_count = len(list(filter(lambda x: x == 2, new_labels)))

        assert class_0_count == min(class_ratios) * row_count
        assert class_1_count == min(class_ratios) * row_count
        assert class_2_count == min(class_ratios) * row_count

    def test_upsample(self):

        row_count = 100000
        class_ratios = [0.7, 0.2, 0.1]
        features, labels = self.generate_imbalance_data(class_ratios, row_count)
        ml_data = MlDataFactory().ml_data_from_numpy(features, labels, indexes=list(range(101, 101+row_count)))

        ml_data_ = UpsampleFC().apply(ml_data)

        new_labels = ml_data_.get_targets()
        class_0_count = len(list(filter(lambda x: x == 0, new_labels)))
        class_1_count = len(list(filter(lambda x: x == 1, new_labels)))
        class_2_count = len(list(filter(lambda x: x == 2, new_labels)))

        assert class_0_count == max(class_ratios) * row_count
        assert class_1_count == max(class_ratios) * row_count
        assert class_2_count == max(class_ratios) * row_count

    def test_upsample_coef(self):

        row_count = 100000
        class_ratios = [0.7, 0.2, 0.1]
        features, labels = self.generate_imbalance_data(class_ratios, row_count)
        ml_data = MlDataFactory().ml_data_from_numpy(features, labels, indexes=list(range(101, 101+row_count)))

        ml_data_ = UpsampleFC(upsample_coef=0.8).apply(ml_data)

        new_labels = ml_data_.get_targets()
        class_0_count = len(list(filter(lambda x: x == 0, new_labels)))
        class_1_count = len(list(filter(lambda x: x == 1, new_labels)))
        class_2_count = len(list(filter(lambda x: x == 2, new_labels)))

        assert class_0_count == int(max(class_ratios) * row_count * 1.0)
        assert class_1_count == int(max(class_ratios) * row_count * 0.8)
        assert class_2_count == int(max(class_ratios) * row_count * 0.8)
