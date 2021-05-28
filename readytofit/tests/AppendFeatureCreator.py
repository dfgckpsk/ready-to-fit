from readytofit.data.FeatureCreator import FeatureCreator
from readytofit.data.MlData import MlData


class AppendFeatureCreator(FeatureCreator):

    def __init__(self, apply_feature, to_column=None, append_value=0):
        FeatureCreator.__init__(self, apply_feature, to_column)
        self.append_value = append_value

    def apply(self, ml_data: MlData):
        features = ml_data.get_features(feature=self.apply_feature)
        features += self.append_value
        ml_data.update_features(features, self.save_feature)
        return ml_data
