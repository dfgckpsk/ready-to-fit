from ..data.FeatureCreator import FeatureCreator
from ..data.MlData import MlData


class AppendFeatureCreator(FeatureCreator):

    def __init__(self, apply_feature, to_column=None, parameters={}):
        FeatureCreator.__init__(self, apply_feature, to_column, parameters=parameters)
        self.append_value = self.parameters.get('append_value')

    def apply(self, ml_data: MlData):
        features = ml_data.get_features(feature=self.apply_feature)
        features += self.append_value
        ml_data.update_features(features, self.save_feature)
        return ml_data
