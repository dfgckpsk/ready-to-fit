from ..tools.logging import logged
from .MlData import MlData
from ..data.CreatorApplyType import CreatorApplyTypeLabel
from ..db.MLDatabaseManager import MLDatabaseManager, MlTsValues


@logged
class LabelCreator:

    def __init__(self, creator_apply_type: CreatorApplyTypeLabel):
        self.run_id = None
        self.database: MLDatabaseManager = None
        self.creator_apply_type = creator_apply_type

    def get(self, data: MlData) -> MlData:
        return data

    def __str__(self):
        not_private_keys = list(filter(lambda x: x[0] != '_', self.__dict__.keys()))
        attr_dict = { your_key: self.__dict__[your_key] for your_key in not_private_keys }
        return f'{self.__class__.__name__}({attr_dict})'

    def dump_labels(self, data: MlData, feature_name: str, indexes_column: str = None):
        if self.database is not None and self.run_id is None:
            self._warning("run id is None, can't dump feature")
        targets = data.get_targets()
        indexes = None
        if indexes_column is not None and indexes in data.feature_names:
            indexes = data.get_features(feature=indexes_column)
        else:
            indexes = data.get_indexes()
        ml_ts_values = MlTsValues(self.run_id, feature_name, float_values=targets, indexes=indexes)
        self.database.insert_ts_values(ml_ts_values)
