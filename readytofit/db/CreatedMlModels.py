from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CreatedMlModels:
    exp_id: str
    run_id_full: datetime
    run_id: datetime
    current_index: datetime
    config: str
    model_id: str

    # metrics
    f1: float = None
    f1_micro: float = None
    f1_macro: float = None
    f1_weighted: float = None
    f1_avg_none: list = field(default_factory=list)
    recall: float = None
    recall_macro_avg: float = None
    recall_weighted_avg: float = None
    recall_avg_none: list = field(default_factory=list)
    precision: float = None
    precision_macro_avg: float = None
    precision_weighted_avg: float = None
    precision_avg_none: list = field(default_factory=list)
    accuracy: float = None
    mse: float = None
    mae: float = None

    # metrics train
    f1_train: float = None
    f1_micro_train: float = None
    f1_macro_train: float = None
    f1_weighted_train: float = None
    f1_avg_none_train: list = field(default_factory=list)
    recall_train: float = None
    recall_macro_avg_train: float = None
    recall_weighted_avg_train: float = None
    recall_avg_none_train: list = field(default_factory=list)
    precision_train: float = None
    precision_macro_avg_train: float = None
    precision_weighted_avg_train: float = None
    precision_avg_none_train: list = field(default_factory=list)
    accuracy_train: float = None
    mse_train: float = None
    mae_train: float = None
    seed: int = None

    data_source_id: str = None
    label_creator_id: str = None
    feature_creator_ids: list = field(default_factory=list)
    data_length: int = None
    features_count: int = None
    split_num: int = None
    label_names: list = field(default_factory=list)
    label_count_train: list = field(default_factory=list)
    label_count_test: list = field(default_factory=list)
    meta: str = None

    def to_dict(self):
        return self.__dict__

    def to_dict_str(self):
        new_dict = {}
        for key in self.__dict__.keys():
            new_dict[key] = str(self.__dict__[key])
        return new_dict

    @staticmethod
    def get_field_names():
        return ['exp_id', 'run_id', 'run_id_full', 'current_index', 'config', 'model_id',

                'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_avg_none', 'recall', 'recall_macro_avg',
                'recall_weighted_avg', 'recall_avg_none', 'precision', 'precision_macro_avg',
                'precision_weighted_avg', 'precision_avg_none', 'accuracy', 'mae', 'mse',

                'f1_train', 'f1_micro_train', 'f1_macro_train', 'f1_weighted_train', 'f1_avg_none_train',
                'recall_train', 'recall_macro_avg_train', 'recall_weighted_avg_train', 'recall_avg_none_train',
                'precision_train', 'precision_macro_avg_train', 'precision_weighted_avg_train',
                'precision_avg_none_train', 'accuracy_train', 'mae_train', 'mse_train',

                'seed', 'data_source_id', 'label_creator_id', 'feature_creator_ids', 'data_length', 'features_count',
                'split_num', 'label_names', 'label_count_train', 'label_count_test']

    @staticmethod
    def from_meta(in_meta: dict):
        meta = in_meta.copy()
        kwargs = {}
        ex_ = CreatedMlModels('', datetime.now(), datetime.now(), 'None', 'None', 0)
        for field_name in CreatedMlModels.get_field_names():
            kwargs[field_name] = meta.get(field_name, getattr(ex_, field_name))
            if field_name in meta.keys():
                del meta[field_name]
        kwargs['meta'] = str(meta) if len(meta) != 0 else None
        created_ml_models = CreatedMlModels(**kwargs)
        return created_ml_models
