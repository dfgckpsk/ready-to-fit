import unittest
from sklearn.datasets import load_iris
from ..tests.TestDataSource import TestDataSource
from ..cv.SLCV import SLCV
from ..data.FeatureCreator import FeatureCreator, CreatorApplyType
from ..data.LabelCreator import LabelCreator, CreatorApplyTypeLabel
from ..data.MlDataFactory import MlDataFactory
from ..db.MLDatabaseManager import MLDatabaseManager
from ..metric.SLMetric import SLMetric
from ..models.LGBMModel import LGMBModel
from ..models.SLModel import SLModel
from ..tests.AppendFeatureCreator import AppendFeatureCreator
from ..tests.AppendLabelCreator import AppendLabelCreator
from ..ModelCreator import ModelCreator
from ..db.clickhouse import Clickhouse
from datetime import datetime


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        data_sample = load_iris()
        self._ml_data = MlDataFactory().ml_data_from_numpy(data_sample.data,
                                                           data_sample.target)
        self._database = Clickhouse.init_database('100.126.106.82:9901:code:SIa/PNNW:backtest')

    def test_simple_model(self):
        model = LGMBModel(parameters={'objective': 'multiclass',
                                      'num_class': 3,
                                      'metric': 'multi_logloss',
                                      'num_iterations': 1})
        model.fit(self._ml_data)
        predictions = model.predict(self._ml_data)
        assert len(predictions) == len(self._ml_data)

    def test_sl_model(self):
        model = SLModel(parameters={'sl_model_name': 'decision_tree_classifier'})
        model.fit(self._ml_data)
        predictions = model.predict(self._ml_data)
        assert len(predictions) == len(self._ml_data)

    def test_factory(self):
        data_sample = load_iris()
        self._ml_data = MlDataFactory().ml_data_from_numpy(data_sample.data,
                                                           data_sample.target)
        assert self._ml_data.features.shape[0] == len(data_sample.data)
        assert len(self._ml_data) == len(data_sample.data)
        assert len(self._ml_data.labels) == len(data_sample.target)
        assert len(self._ml_data.get_features([1,2,3,4])) == 4
        assert len(self._ml_data.get_targets([1,2,3,4])) == 4

    def test_model_creator(self):
        feature_creators = [FeatureCreator(apply_feature='', creator_apply_type=CreatorApplyType.BeforeTrainBeforeLabel)]
        label_creator = LabelCreator(CreatorApplyTypeLabel.BeforeTrain)
        ml_model_manager = MLDatabaseManager(self._database)
        run_id = datetime.utcnow()
        model_creator = ModelCreator('test_model_creator',
                                     'lgbm',
                                     label_creator,
                                     feature_creators,
                                     data_source=TestDataSource(run_id, ml_database_manager=ml_model_manager),
                                     parameters={'objective': 'multiclass',
                                                 'num_class': 3,
                                                 'metric': 'multi_logloss',
                                                 'num_iterations': 1},
                                     ml_database_manager=ml_model_manager,
                                     run_id=run_id)
        model_creator.train()

        results = ml_model_manager.get_run_result(run_id)
        print(results)
        assert len(results) == 1

    def test_model_creator_validation(self):
        feature_creators = [FeatureCreator(apply_feature='', creator_apply_type=CreatorApplyType.BeforeTrainBeforeLabel)]
        label_creator = LabelCreator(CreatorApplyTypeLabel.BeforeTrain)
        ml_model_manager = MLDatabaseManager(self._database)
        run_id = datetime.utcnow()
        model_creator = ModelCreator('test_model_creator_validation',
                                     'lgbm',
                                     label_creator,
                                     feature_creators,
                                     data_source=TestDataSource(run_id, ml_database_manager=ml_model_manager),
                                     metrics=[SLMetric('accuracy')],
                                     parameters={'objective': 'multiclass',
                                                 'num_class': 3,
                                                 'metric': 'multi_logloss',
                                                 'num_iterations': 1},
                                     ml_database_manager=ml_model_manager,
                                     run_id=run_id)
        model_creator.validation(SLCV('kfold'))

        results = ml_model_manager.get_run_result(run_id)
        assert len(results) == 5

    def test_model_creator_validation_sl(self):
        feature_creators = [FeatureCreator(apply_feature='', creator_apply_type=CreatorApplyType.BeforeTrainBeforeLabel)]
        label_creator = LabelCreator(CreatorApplyTypeLabel.BeforeTrain)
        run_id = datetime.utcnow()
        model_creator = ModelCreator('test_model_creator_validation',
                                     'slmodel',
                                     label_creator,
                                     feature_creators,
                                     data_source=TestDataSource(run_id),
                                     metrics=[SLMetric('accuracy')],
                                     parameters={'sl_model_name': 'decision_tree_classifier'})
        model_creator.validation(SLCV('kfold'))

    def test_label_creator(self):
        append_value = 3
        before_targets = self._ml_data.get_targets()
        self._ml_data = AppendLabelCreator(append_value=append_value).get(self._ml_data)
        after_targets = self._ml_data.get_targets()
        assert (before_targets + append_value == after_targets).all()

    def test_feature_creator(self):
        append_value = 3
        before_features = self._ml_data.get_features(feature='feature_1')
        self._ml_data = AppendFeatureCreator(parameters=dict(append_value=append_value), apply_feature='feature_1').apply(self._ml_data)
        after_features = self._ml_data.get_features(feature='feature_1')
        assert (before_features + append_value == after_features).all()

        append_value = 5
        before_features = self._ml_data.get_features(feature='feature_2')
        self._ml_data = AppendFeatureCreator(parameters=dict(append_value=append_value),
                                             apply_feature='feature_2',
                                             to_column='feature_4').apply(self._ml_data)
        after_features = self._ml_data.get_features(feature='feature_4')
        assert (before_features + append_value == after_features).all()
