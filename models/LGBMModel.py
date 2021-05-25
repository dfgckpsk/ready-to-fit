from .Model import Model, MlData
import lightgbm as lgb
import numpy as np
from tools.logging import logged


@logged
class LGMBModel(Model):

    """
    Class for handle lightgbm model
    """

    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)

        self.model_object = None
        self.validation_obj = self.parameters.get('validation_cv_obj')
        if 'validation_cv_obj' in self.parameters:
            del self.parameters['validation_cv_obj']

    # def log_validation(self, eval_result):
    #
    #     train_iter_count = 0
    #     for key in eval_result.keys():
    #         for metric_key in eval_result[key].keys():
    #             name = key + '_' + metric_key
    #             values = eval_result[key][metric_key]
    #             train_iter_count = len(values)
    #             # for i, v in enumerate(values):
    #             #     self.logger.log_metric(name, v, i)
    #     # self.logger.log_param("early_stopping_round_number", train_iter_count)

    def split_indexes_for_incremental(self, ind, ml_data):

        """
        Generate indexes for inc learning
        :param ind:
        :param ml_data:
        :return:
        """

        incremental_size = self.parameters.get('incremental_size')

        if ind is None:
            ind = ml_data.get_indexes()

        if incremental_size is None:
            return False, [ind]
        else:
            indexes = np.array_split(ind, len(ind) // incremental_size + 1)
            return True, indexes

    def fit(self, ml_data: MlData):

        self._debug(f'Start train {self} on data with len {len(ml_data)} and features count {len(ml_data.feature_names)}')
        validation_indexes = None
        if self.validation_obj is not None:
            validation_splits = list(self.validation_obj.split(ml_data))
            if len(validation_splits) > 1:
                self._critical(f'Validation split len should be equal 1, but equal {len(validation_splits)}')
            train_indexes, validation_indexes = validation_splits[0]
        else:
            train_indexes = ml_data.get_indexes()

        if validation_indexes is not None:
            self._debug(f'All data splitted on train and validation. All data - {len(ml_data)}, '
                        f'train - {len(train_indexes)}, '
                        f'validation - {len(validation_indexes)}')

        inc_available, train_indexes = self.split_indexes_for_incremental(train_indexes, ml_data)

        eval_results = []
        model_obj = None
        validation_features = ml_data.get_features(validation_indexes) if validation_indexes is not None else None
        validation_labels = ml_data.get_targets(validation_indexes) if validation_indexes is not None else None
        validation_weights = ml_data.get_weights(validation_indexes) if validation_indexes is not None else None

        for ind in train_indexes:
            model_obj, eval_result = self.fit_step(ml_data.get_features(ind),
                                                   ml_data.get_targets(ind),
                                                   model_object=model_obj,
                                                   weights=ml_data.get_weights(ind),
                                                   validation_features=validation_features,
                                                   validation_labels=validation_labels,
                                                   validation_weights=validation_weights)
            eval_results.append(eval_result)
        return {}

    def fit_step(self, features, labels, model_object, weights=None, validation_features=None, validation_labels=None, validation_weights=None):

        eval_result = {}

        # TODO weight support
        lgb_eval = None
        lgb_train = lgb.Dataset(features,
                                labels.reshape(-1),
                                free_raw_data=False,
                                weight=weights)

        if validation_features is not None and validation_labels is not None:
            lgb_eval = lgb.Dataset(validation_features,
                                   validation_labels.reshape(-1),
                                   reference=lgb_train,
                                   free_raw_data=False,
                                   weight=validation_weights)

        self.model_object = lgb.train(self.parameters,
                                      lgb_train,
                                      init_model=model_object,
                                      valid_sets=lgb_eval,
                                      early_stopping_rounds=self.parameters.get('early_stopping_rounds'),
                                      evals_result=eval_result)

        return self.model_object, eval_result

    def predict(self, ml_data: MlData):

        pred = self.predict_step(self.model_object.predict,
                                 ml_data)
        if len(pred) == 0:
            return np.array([])
        # TODO fix num_iteration
        if self.parameters.get('objective') == 'multiclass':
            pred = np.argmax(pred, axis=1)

        return pred.reshape(-1, 1)

    def predict_proba(self, ml_data):

        return self.predict(ml_data)

    def predict_step(self, predict_method, ml_data):

        """
        Predict with incremental support
        :param predict_method:
        :param ml_data:
        :return:
        """

        in_indexes = ml_data.get_indexes()
        incremental_available, indexes = self.split_indexes_for_incremental(in_indexes, ml_data)

        if not incremental_available:
            pred = predict_method(ml_data.get_features(in_indexes))
            return pred

        else:

            pred_arr = []
            for ind in indexes:
                pred = predict_method(ml_data.features(ind))
                pred_arr.append(pred)
            pred = np.concatenate(pred_arr)

            return pred

    def dump(self, path, model_name):

        """
        Dump model
        :param path:
        :param model_name:
        :return:
        """

        self.model_object.save_model(path + '/' + model_name + '.model')

    def load(self, path):

        self.model_object = lgb.Booster(model_file=path)
