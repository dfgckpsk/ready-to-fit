from .cv.CVBase import CVBase
from .data.DataSource import DataSource
from .data.FeatureCreator import FeatureCreator
from .data.LabelCreator import LabelCreator, CreatorApplyTypeLabel
from .data.MlData import MlData
from .data.MlDataFactory import MlDataFactory
from .db.CreatedMlModels import CreatedMlModels
from .db.MLDatabaseManager import MLDatabaseManager, MlTsValues
from .metric.MetricBase import MetricBase
from .models.ModelFactory import ModelFactory
from typing import List, Optional
from .tools import logged
from .db.MlParameterFactory import MLParameterFactory, MlParameterType

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
                 data_source: DataSource,
                 metrics: Optional[List[MetricBase]] = None,
                 parameters: dict = None,
                 ml_database_manager: MLDatabaseManager = None,
                 run_id: datetime = None):
        self.model_name = model_name
        self.label_creator = label_creator
        self.feature_creators = feature_creators
        self.metrics = metrics
        self.data_source = data_source
        self.parameters = parameters
        self.exp_id = exp_id
        self.ml_database_manager = ml_database_manager
        self.run_id = run_id
        self.seed = self.parameters.get('seed')
        self.validate_train = self.parameters.get('validate_train', True)
        if self.run_id is None:
            self.run_id = datetime.utcnow()

    def train(self):
        ml_data = self.data_source.generate_data()
        try:
            model, meta, _ = self._train(ml_data)
            self._log_meta(meta)
            return model, meta
        except BaseException as e:
            self._error(e)
            return None, None

    def validation(self, cv: CVBase, return_models: bool = False):
        split_num = 0
        models = []
        ml_data = self.data_source.generate_data()
        ml_data = self._prepare_data(ml_data, False, once_all_data=True)
        for train_indexes, test_indexes in cv.split(ml_data):
            train_data = MlDataFactory().ml_data_from_ml_data(ml_data,
                                                              train_indexes)
            model, meta, prepared_train = self._train(train_data, split_num)
            models.append(model)
            test_data = MlDataFactory().ml_data_from_ml_data(ml_data,
                                                             test_indexes)
            test_data = self._prepare_data(test_data, preparing_for_validation=True, skip_label_creator=False)
            meta = self._calculate_labels_count(meta, test_data, False)

            self._debug(f'Validation number {split_num} with length; '
                        f'train_data:{len(train_data)}, '
                        f'test_data:{len(test_data)}, '
                        f'train_indexes:{len(train_indexes)};{train_indexes[0]}, '
                        f'test_indexes:{len(test_indexes)};{test_indexes[0]}.')
            meta['split_num'] = split_num
            meta = self._dump_metrics(model, prepared_train, test_data, meta, split_num)
            self._log_meta(meta)
            split_num += 1
        if return_models:
            return models

    def _prepare_labels(self, new_ml_data):
        if self.label_creator is not None:
            self.label_creator.run_id = self.run_id
            self.label_creator.database = self.ml_database_manager
            new_ml_data = self.label_creator.get(new_ml_data)
        return new_ml_data

    def _prepare_features(self,
                          new_ml_data,
                          preparing_for_validation,
                          skip_label_creator,
                          after_label: bool,
                          once_all_data: bool):
        for featore_creator in list(filter(lambda x: after_label == x.apply_after_label and
                                                     (not x.use_just_for_train or x.use_just_for_train and not preparing_for_validation)
                                                     and once_all_data == x.once_all_data,
                                           self.feature_creators)):
            featore_creator.run_id = self.run_id
            featore_creator.database = self.ml_database_manager
            new_ml_data = featore_creator.apply(new_ml_data)
            print(f'Applied {"" if not after_label else "after label"} {featore_creator}, '
                        f'data_len {len(new_ml_data)}. '
                        f'skip_label_creator={skip_label_creator}, feature len={len(new_ml_data.feature_names)},'
                  f'once_all_data={once_all_data}, after_label={after_label}, skip_label_creator={skip_label_creator},'
                  f'featore_creator={featore_creator}')
        return new_ml_data

    def _prepare_data(self, ml_data: MlData,
                      preparing_for_validation: bool,
                      skip_label_creator: bool = False,
                      once_all_data: bool = False):
        new_ml_data = ml_data.__copy__()
        new_ml_data = self._prepare_features(new_ml_data, preparing_for_validation, skip_label_creator, False, once_all_data)

        if once_all_data == (self.label_creator.creator_apply_type == CreatorApplyTypeLabel.OnceOnAllData) and not skip_label_creator:
            new_ml_data = self._prepare_labels(new_ml_data)
        new_ml_data = self._prepare_features(new_ml_data, preparing_for_validation, skip_label_creator, True, once_all_data)

        self._debug(f'Prepare data canceled, data_len {len(new_ml_data)}')
        return new_ml_data

    def _set_seeds(self, custom_seed=None):
        seed = random.randint(0, 100000) if custom_seed is None else custom_seed
        np.random.seed(seed)
        random.seed(seed)
        return seed

    def _calculate_labels_count(self, meta: dict, ml_data: MlData, train):
        if ml_data.label_names is not None:
            targets = ml_data.get_targets()
            _, counts = np.unique(targets, return_counts=True)
            key = 'label_count_train' if train else 'label_count_test'
            meta[key] = counts
        return meta

    def _fill_meta(self, in_meta: dict, ml_data: MlData):
        in_meta['run_id'] = self.run_id
        in_meta['run_id_full'] = self.run_id
        in_meta['exp_id'] = self.exp_id
        in_meta['config'] = str(clear_parameters_dict_string(self.parameters))
        in_meta['data_source_id'] = str(self.data_source.__class__.__name__)
        in_meta['label_creator_id'] = str(self.label_creator) if self.label_creator is not None else None
        in_meta['label_names'] = ml_data.label_names if ml_data.label_names is not None else []
        if in_meta['label_names'] is None:
            in_meta['label_names'] = []
        in_meta['feature_creator_ids'] = list(map(str, self.feature_creators))
        return in_meta

    def _log_meta(self, in_meta: dict):
        if self.ml_database_manager is not None:
            created_ml_model = CreatedMlModels.from_meta(in_meta)
            self.ml_database_manager.insert_experiment(created_ml_model)

    def _log_parameters(self, parameters: dict, split_num: int = None):

        if self.ml_database_manager is not None:
            ml_parameters = MLParameterFactory.from_dict(parameters, self.run_id, MlParameterType.Model, split_num)
            self.ml_database_manager.insert_parameters(ml_parameters)

    def _train(self, ml_data: MlData, split_num: int = None):
        seed = self._set_seeds(self.seed)
        new_ml_data = self._prepare_data(ml_data, preparing_for_validation=False, once_all_data=False)
        model = ModelFactory().get_model(self.model_name, self.parameters)
        self._log_parameters(model.parameters, split_num)
        meta = model.fit(new_ml_data)
        # if current_index is not None:
        #     meta['current_index'] = current_index.get_datetime()
        meta['current_index'] = self.run_id
        meta['features_count'] = len(new_ml_data.feature_names)
        meta['data_length'] = len(new_ml_data)
        meta['seed'] = seed
        meta['model_id'] = str(model)
        meta = self._fill_meta(meta, new_ml_data)
        meta = self._calculate_labels_count(meta, new_ml_data, True)
        return model, meta, new_ml_data

    def _dump_ts_metrics(self, data: MlData, values, value_name, split_num):
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

    def _dump_metrics(self, model, prepared_train: MlData, test_data: MlData, meta: dict, split_num: int):
        if self.metrics is not None:
            predicted = model.predict(test_data)
            metric_results_train = {}
            if self.validate_train:
                train_predicted = model.predict(prepared_train)
                metric_results_train = dict(map(lambda metric:
                                                (str(metric) + '_train',
                                                 metric.calculate(prepared_train.get_targets(),
                                                                  train_predicted)), self.metrics))
                self._dump_ts_metrics(prepared_train, train_predicted, 'predicted_train', split_num)
                self._dump_ts_metrics(prepared_train, prepared_train.get_targets(), 'labels_train', split_num)
            metric_results = dict(map(lambda metric:
                                      (str(metric),
                                       metric.calculate(test_data.get_targets(), predicted)), self.metrics))
            self._dump_ts_metrics(test_data, predicted, 'predicted', split_num)
            self._dump_ts_metrics(test_data, test_data.get_targets(), 'labels', split_num)
            meta = {**meta, **metric_results, **metric_results_train}
        return meta
