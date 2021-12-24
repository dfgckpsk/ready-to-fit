from ..models.Model import Model
from ..data.MlData import MlData
from sklearn.utils import _joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from ..tools.logging import logged


@logged
class SLModel(Model):

    """
    Class for handle scikit-learn model
    """

    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)

        self.model_name = self.parameters.get('sl_model_name')
        if 'sl_model_name' in self.parameters.keys():
            del self.parameters['sl_model_name']
        if 'seed' in self.parameters.keys():
            del self.parameters['seed']
        self.model_class = None
        if self.model_name == 'decision_tree_classifier':
            self.model_class = DecisionTreeClassifier
        elif self.model_name == 'random_forest_classifier':
            self.model_class = RandomForestClassifier
        elif self.model_name.lower() == 'pca':
            self.model_class = PCA
        self.model_object = self.model_class(**self.parameters)

    def fit(self, ml_data: MlData):

        """
        Fit
        :param ml_data:
        :return:
        """
        Model.fit(self, ml_data)
        self._debug(f'Start train {self} on data with len {len(ml_data)}')
        self.model_object.fit(ml_data.get_features(),
                              ml_data.get_targets(),
                              sample_weight=ml_data.get_weights())
        return {}

    def predict(self, ml_data: MlData):

        """
        Predict
        :param ml_data:
        :return:
        """

        if self.model_name.lower() == 'pca':
            return self.model_object.components_
        return self.model_object.predict(ml_data.get_features(feature=self.meta.get('feature_names')))

    def predict_proba(self, ml_data):

        """
        Predict proba
        :param ml_data:
        :return:
        """

        return self.model_object.predict_proba(ml_data.get_features(feature=self.meta.get('feature_names')))

    def dump(self, path, model_name):

        """
        Dump model
        :param path:
        :return:
        """
        Model.dump(self, path, model_name)

        _joblib.dump(self.model_object, path + '/' + model_name + '.model')

    def load(self, path):
        Model.load(self, path)
        self.model_object = _joblib.load(path)

    def __str__(self):
        return self.__class__.__name__ + '_' + self.model_name
