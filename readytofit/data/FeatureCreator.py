from readytofit.data.MlData import MlData
from readytofit.db.MlTsValues import MlTsValues
from readytofit.db.MLDatabaseManager import MLDatabaseManager
from readytofit.tools.logging import logged
from readytofit.data.CreatorApplyType import CreatorApplyType


@logged
class FeatureCreator:

    def __init__(self, apply_feature=None, to_column=None, use_just_for_train=False, creator_apply_type: CreatorApplyType = CreatorApplyType.BeforeTrainBeforeLabel):
        self.apply_feature = apply_feature
        self.save_feature = to_column
        self.use_just_for_train = use_just_for_train
        if self.save_feature is None:
            self.save_feature = self.apply_feature
        self.run_id = None
        self.database: MLDatabaseManager = None
        self.creator_apply_type = creator_apply_type

    @property
    def apply_after_label(self):
        return self.creator_apply_type in (CreatorApplyType.BeforeTrainAfterLabel, CreatorApplyType.OnceOnAllDataAfterLabel)

    @property
    def once_all_data(self):
        return self.creator_apply_type in (CreatorApplyType.OnceOnAllDataAfterLabel, CreatorApplyType.OnceOnAllDataBeforeLabel)

    def apply(self, ml_data: MlData):
        return ml_data

    def __str__(self):
        not_private_keys = list(filter(lambda x: x[0] != '_', self.__dict__.keys()))
        attr_dict = { your_key: self.__dict__[your_key] for your_key in not_private_keys }
        return f'{self.__class__.__name__}({attr_dict})'

    def left_columns(self, current_columns):
        left_columns = []
        for column in current_columns:
            if column not in self.apply_feature:
                left_columns.append(column)
        return left_columns

    def dump_column(self, data: MlData, feature_name: str, indexes_column: str = None):
        if self.database is not None and self.run_id is None:
            self._warning("run id is None, can't dump feature")
        features = data.get_features(feature=feature_name)
        indexes = None
        if indexes_column is not None and indexes in data.feature_names:
            indexes = data.get_features(feature=indexes_column)
        else:
            indexes = data.get_indexes()
        ml_ts_values = MlTsValues(self.run_id, feature_name, float_values=features, indexes=indexes)
        self.database.insert_ts_values(ml_ts_values)
