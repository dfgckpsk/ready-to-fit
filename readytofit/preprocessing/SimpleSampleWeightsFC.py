from readytofit.data.FeatureCreator import FeatureCreator, MlData
import numpy as np
import pandas as pd


class SimpleSampleWeightsFC(FeatureCreator):

    def apply(self, ml_data: MlData):
        targets = ml_data.get_targets()
        unique, counts = np.unique(targets, return_counts=True)

        count_sum = len(ml_data)
        counts_ratio = list(map(lambda x: count_sum - x, counts))
        counts_ratio_sum = sum(counts_ratio)
        counts_ratio = list(map(lambda x: x/counts_ratio_sum, counts_ratio))

        weight_by_label = dict(zip(unique, counts_ratio))
        weights = list(map(lambda x: weight_by_label[x], targets))
        weights = pd.Series(weights, index=ml_data.get_indexes())
        ml_data.update_weights(weights)
        return ml_data
