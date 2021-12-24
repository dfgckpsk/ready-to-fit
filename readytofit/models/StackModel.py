from .Model import Model, MlData
from ..models.SimpleModelFactory import SimpleModelFactory
from ..data.MlDataFactory import MlDataFactory
from ..tools.logging import logged
from typing import List


@logged
class StackModel(Model):

    """
    parameters:
    feature_creator/models: dict with keys - numbers of layer and values - objects list
    label_creator/split_objects/model_parameters: dict with keys - numbers of layer and values - objects

    split_objects with key - 0 split data for 0 and 1 layer, with key - 1 for 1 and 2, etc.

    as result
    self.trained_models - dict with key - layer number, values - list of models
    """

    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)

        self.models = self.parameters.get('models')
        self.model_parameters = self.parameters.get('model_parameters')
        self.feature_creators = self.parameters.get('feature_creators')
        self.label_creator = None
        self.split_objects = self.parameters.get('split_objects')
        self.layers = self.models.keys()

        if not len(self.split_objects) + 1 == len(self.models):
            self._critical(f'split objects have keys - {self.split_objects.keys()},'
                           f'but model have keys - {self.models.keys()}. '
                           f'model layers should be on one more than split object layers')

        self.inited_models = {}

    def init_models(self):
        if self.models is None:
            self._critical('Models is None')
        factory = SimpleModelFactory()
        inited_models = {}
        for layer in self.models.keys():
            inited_models[layer] = []

            parameters = self.model_parameters.get(layer, {})
            for model_name in self.models[layer]:
                model = factory.get_model(model_name, parameters)
                inited_models[layer].append(model)
        return inited_models

    def split_data(self, ml_data: MlData):

        # TODO: should be tested for more than 1 layer
        data_by_layers = {}
        factory = MlDataFactory()
        for layer in self.split_objects.keys():
            if layer == 0:
                ml_data_ = ml_data.__copy__()
            else:
                ml_data_ = data_by_layers[layer]
            ind1, ind2 = list(self.split_objects[layer].split(ml_data))[0]
            data_by_layers[layer] = factory.ml_data_from_ml_data(ml_data_,
                                                                 ind1)
            data_by_layers[layer+1] = factory.ml_data_from_ml_data(ml_data_,
                                                                   ind2)
        return data_by_layers

    def inference_layer_data(self, ml_data: MlData, models: List[Model]) -> MlData:
        inference_results = []
        new_ml_data = ml_data.__copy__()
        for i, model in enumerate(models):
            inference_result = model.predict(ml_data)
            inference_results.append(inference_result)
            new_ml_data.update_features(inference_result, f'inference_{i}')
        return new_ml_data

    def prepare_data(self, ml_data: MlData, layer: int, preparing_for_validation: bool, skip_label_creator: bool = False):
        new_ml_data = ml_data.__copy__()
        feature_creators = self.feature_creators.get(layer, [])
        for featore_creator in list(filter(lambda x: not x.apply_after_label and
                                                     (not x.use_just_for_train or x.use_just_for_train and
                                                      not preparing_for_validation),
                                           feature_creators)):
            new_ml_data = featore_creator.apply(new_ml_data)
            self._debug(f'Applied {featore_creator}, data_len {len(new_ml_data)}. '
                        f'skip_label_creator={skip_label_creator}, feature len={len(new_ml_data.feature_names)}')
        if self.label_creator is not None and not skip_label_creator:
            new_ml_data = self.label_creator.get(new_ml_data)
        for featore_creator in list(filter(lambda x: x.apply_after_label and
                                                     (not x.use_just_for_train or x.use_just_for_train and
                                                      not preparing_for_validation),
                                           feature_creators)):
            self._debug(f'Applied after label {featore_creator}, '
                        f'data_len {len(new_ml_data)}. skip_label_creator={skip_label_creator}, '
                        f'feature len={len(new_ml_data.feature_names)}')
            new_ml_data = featore_creator.apply(new_ml_data)
        self._debug(f'Prepare data canceled, data_len {len(new_ml_data)}')
        return new_ml_data

    def fit(self, ml_data: MlData):

        self.inited_models = self.init_models()
        data_by_layers = self.split_data(ml_data)
        for layer in self.layers:

            if layer == 0:
                for inited_model in self.inited_models[layer]:
                    data = self.prepare_data(data_by_layers.get(layer, ml_data), False, False)
                    inited_model.fit(data)
                    # self.trained_models[layer].append(trained_model)
            else:
                data = self.inference_layer_data(data_by_layers.get(layer, ml_data), self.inited_models[layer-1])
                data = self.prepare_data(data, layer, False, False)
                for inited_model in self.inited_models[layer]:
                    inited_model.fit(data)
        return {}

    def predict(self, ml_data: MlData):

        data = ml_data.__copy__()
        for layer in self.inited_models.keys():
            data = self.inference_layer_data(data, self.inited_models[layer])

        return data.get_features(feature='inference_0')
