from unittest import TestCase
from ..models.ModelFactory import ModelFactory
from ..tests.AppendFeatureCreator import AppendFeatureCreator
from sklearn.datasets import load_iris
from ..data.MlDataFactory import MlDataFactory
from ..cv.SLCV import SLCV
from ..models.SLModel import SLModel
from ..models.LGBMModel import LGMBModel


class TestStackModel(TestCase):

    # TODO: added tests for more than 2 layers

    def test(self):

        parameters = {}
        data_sample = load_iris()
        self._ml_data = MlDataFactory().ml_data_from_numpy(data_sample.data,
                                                           data_sample.target)
        feature_creators = {0: [AppendFeatureCreator(apply_feature='feature_0', append_value=1)]}
        split_objects = {0: SLCV(class_name='kfold', n_splits=1)}

        parameters_layer_0 = {'sl_model_name': 'random_forest_classifier'}
        parameters_layer_1 = {'task': 'train',
                              'num_class': 3,
                              'boosting_type': 'gbdt',
                              'objective': 'multiclass'}
        models = {0: ['slmodel']*5,
                  1: ['lgbm']}
        model_parameters = {0: parameters_layer_0,
                            1: parameters_layer_1}

        parameters['feature_creators'] = feature_creators
        parameters['split_objects'] = split_objects
        parameters['models'] = models
        parameters['model_parameters'] = model_parameters

        stack_model = ModelFactory().get_model('stack_model', parameters)

        inited_models = stack_model.init_models()
        assert all(map(lambda x: type(x) == SLModel, inited_models[0]))
        assert len(inited_models[0]) == 5
        assert all(map(lambda x: type(x) == LGMBModel, inited_models[1]))
        assert len(inited_models[1]) == 1

        splitted_data = stack_model.split_data(ml_data=self._ml_data)
        assert (splitted_data[0].get_indexes() == self._ml_data.get_indexes()[:75]).all()
        assert (splitted_data[1].get_indexes() == self._ml_data.get_indexes()[75:]).all()

        stack_model.fit(self._ml_data)