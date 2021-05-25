from cv.CVBase import CVBase
from data.FeatureCreator import FeatureCreator
from data.LabelCreator import LabelCreator
from data.MlData import MlData
from data.MlDataFactory import MlDataFactory
from db.CreatedMlModels import CreatedMlModels
from db.MLDatabaseManager import MLDatabaseManager, MlTsValues
from metric.MetricBase import MetricBase
from models.ModelFactory import ModelFactory
from typing import List, Optional
from tools import logged

from datetime import datetime
import numpy as np
import random


def clear_parameters_dict_string(in_dict: dict):
    cleared = {}
    for k in in_dict.keys():
        v = in_dict[k]
        if type(v) == dict:
            cleared[k] = clear_parameters_dict_string(v)
        elif type(v) == list:
            cleared[k] = ', '.join(str(e) for e in v)
        else:
            cleared[k] = str(v)
    return cleared


@logged
class ModelCreator:

    def __init__(self,
                 exp_id: str,
                 model_name: str,
                 label_creator: LabelCreator,
                 feature_creators: List[FeatureCreator],
                 metrics: Optional[List[MetricBase]] = None,
                 parameters: dict = None,
                 ml_database_manager: MLDatabaseManager = None,
                 run_id: datetime = None):
        self.model_name = model_name
        self.label_creator = label_creator
        self.feature_creators = feature_creators
        self.metrics = metrics
        self.parameters = parameters
        self.exp_id = exp_id
        self.ml_database_manager = ml_database_manager
        self.run_id = run_id
        self.seed = self.parameters.get('seed')
        self.validate_train = self.parameters.get('validate_train', True)
        if self.run_id is None:
            self.run_id = datetime.utcnow()

    def prepare_data(self, ml_data: MlData, preparing_for_validation: bool, skip_label_creator: bool = False):
        new_ml_data = ml_data.__copy__()
        for featore_creator in list(filter(lambda x: not x.apply_after_label and
                                                     (not x.use_just_for_train or x.use_just_for_train and
                                                      not preparing_for_validation),
                                           self.feature_creators)):
            featore_creator.run_id = self.run_id
            featore_creator.database = self.ml_database_manager
            new_ml_data = featore_creator.apply(new_ml_data)
            self._debug(f'Applied {featore_creator}, data_len {len(new_ml_data)}. '
                        f'skip_label_creator={skip_label_creator}, feature len={len(new_ml_data.feature_names)}')
        if self.label_creator is not None and not skip_label_creator:
            self.label_creator.run_id = self.run_id
            self.label_creator.database = self.ml_database_manager
            new_ml_data = self.label_creator.get(new_ml_data)
        for featore_creator in list(filter(lambda x: x.apply_after_label and
                                                     (not x.use_just_for_train or x.use_just_for_train and
                                                      not preparing_for_validation),
                                           self.feature_creators)):
            featore_creator.run_id = self.run_id
            featore_creator.database = self.ml_database_manager
            self._debug(f'Applied after label {featore_creator}, '
                        f'data_len {len(new_ml_data)}. skip_label_creator={skip_label_creator}, '
                        f'feature len={len(new_ml_data.feature_names)}')
            new_ml_data = featore_creator.apply(new_ml_data)
        self._debug(f'Prepare data canceled, data_len {len(new_ml_data)}')
        return new_ml_data

    def set_seeds(self, custom_seed=None):
        seed = random.randint(0, 100000) if custom_seed is None else custom_seed
        np.random.seed(seed)
        random.seed(seed)
        return seed

    def calculate_labels_count(self, meta: dict, ml_data: MlData, train):
        if ml_data.label_names is not None:
            targets = ml_data.get_targets()
            _, counts = np.unique(targets, return_counts=True)
            key = 'label_count_train' if train else 'label_count_test'
            meta[key] = counts
        return meta

    def fill_meta(self, in_meta: dict, ml_data: MlData):
        in_meta['run_id'] = self.run_id
        in_meta['exp_id'] = self.exp_id
        in_meta['config'] = str(clear_parameters_dict_string(self.parameters))
        in_meta['label_creator_id'] = str(self.label_creator) if self.label_creator is not None else None
        in_meta['label_names'] = ml_data.label_names if ml_data.label_names is not None else []
        if in_meta['label_names'] is None:
            in_meta['label_names'] = []
        in_meta['feature_creator_ids'] = list(map(str, self.feature_creators))
        return in_meta

    def log_meta(self, in_meta: dict):
        if self.ml_database_manager is not None:
            created_ml_model = CreatedMlModels.from_meta(in_meta)
            self.ml_database_manager.insert_experiment(created_ml_model)

    def train(self, ml_data: MlData):
        try:
            model, meta, _ = self._train(ml_data)
            self.log_meta(meta)
            return model, meta
        except BaseException as e:
            self._error(e)
            return None, None

    def _train(self, ml_data: MlData):
        seed = self.set_seeds(self.seed)
        new_ml_data = self.prepare_data(ml_data, preparing_for_validation=False)
        model = ModelFactory().get_model(self.model_name, self.parameters)
        meta = model.fit(new_ml_data)
        # if current_index is not None:
        #     meta['current_index'] = current_index.get_datetime()
        meta['features_count'] = len(new_ml_data.feature_names)
        meta['data_length'] = len(new_ml_data)
        meta['seed'] = seed
        meta['model_id'] = str(model)
        meta = self.fill_meta(meta, new_ml_data)
        meta = self.calculate_labels_count(meta, new_ml_data, True)
        return model, meta, new_ml_data

    def dump_ts_metrics(self, data: MlData, values, value_name, split_num):
        if self.run_id is None or self.ml_database_manager is None:
            return
        if 'timestamp' in data.feature_names:
            indexes = data.get_features(feature='timestamp')
        else:
            indexes = data.get_indexes()
        ts_values = MlTsValues(run_id=self.run_id,
                               split_num=split_num,
                               value_name=value_name,
                               indexes=indexes,
                               float_values=values)
        self.ml_database_manager.insert_ts_values(ts_values)

    def dump_metrics(self, model, prepared_train: MlData, test_data: MlData, meta: dict, split_num: int):
        if self.metrics is not None:
            predicted = model.predict(test_data)
            metric_results_train = {}
            if self.validate_train:
                train_predicted = model.predict(prepared_train)
                metric_results_train = dict(map(lambda metric:
                                                (str(metric) + '_train',
                                                 metric.calculate(prepared_train.get_targets(),
                                                                  train_predicted)), self.metrics))
                self.dump_ts_metrics(prepared_train, train_predicted, 'predicted_train', split_num)
                self.dump_ts_metrics(prepared_train, prepared_train.get_targets(), 'labels_train', split_num)
            metric_results = dict(map(lambda metric:
                                      (str(metric),
                                       metric.calculate(test_data.get_targets(), predicted)), self.metrics))
            self.dump_ts_metrics(test_data, predicted, 'predicted', split_num)
            self.dump_ts_metrics(test_data, test_data.get_targets(), 'labels', split_num)
            meta = {**meta, **metric_results, **metric_results_train}
        return meta

    def validation(self, ml_data: MlData, cv: CVBase, return_models: bool = False):
        split_num = 0
        models = []
        for train_indexes, test_indexes in cv.split(ml_data):
            train_data = MlDataFactory().ml_data_from_ml_data(ml_data,
                                                              train_indexes)
            model, meta, prepared_train = self._train(train_data)
            models.append(model)
            test_data = MlDataFactory().ml_data_from_ml_data(ml_data,
                                                             test_indexes)
            test_data = self.prepare_data(test_data, preparing_for_validation=True, skip_label_creator=False)
            meta = self.calculate_labels_count(meta, test_data, False)

            self._debug(f'Validation number {split_num} with length; '
                        f'train_data:{len(train_data)}, '
                        f'test_data:{len(test_data)}, '
                        f'train_indexes:{len(train_indexes)};{train_indexes[0]}, '
                        f'test_indexes:{len(test_indexes)};{test_indexes[0]}.')
            meta['split_num'] = split_num
            meta = self.dump_metrics(model, prepared_train, test_data, meta, split_num)
            self.log_meta(meta)
            split_num += 1
        if return_models:
            return models
