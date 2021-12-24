from ..data.FeatureCreator import FeatureCreator, MlData, CreatorApplyType
from ..data.MlDataFactory import MlDataFactory
import pandas as pd
from ..tools.logging import logged


@logged
class LeaveLabelsFC(FeatureCreator):

    def __init__(self, apply_feature=None, to_column=None, use_just_for_train=False, creator_apply_type: CreatorApplyType = CreatorApplyType.BeforeTrain, leave_labels=None, new_label_names=None):
        FeatureCreator.__init__(self, apply_feature, to_column, use_just_for_train, creator_apply_type)
        self.leave_labels = leave_labels
        self.new_label_names = new_label_names

    def apply(self, ml_data: MlData):
        if self.leave_labels is None:
            self._warning('leave labels is None')
            return ml_data
        if self.new_label_names is not None and len(self.new_label_names) != len(self.leave_labels):
            self._critical(f'new_label_names ({len(self.new_label_names)}) is not equal to leave_labels '
                           f'({len(self.leave_labels)})')
            return ml_data
        labels = ml_data.get_targets()
        labels = pd.Series(labels, index=ml_data.get_indexes())
        labels = labels[labels.isin(self.leave_labels)]
        new_ml_data = MlDataFactory().ml_data_from_ml_data(ml_data, labels.index)
        if self.new_label_names is not None:
            new_ml_data.label_names = self.new_label_names
        return new_ml_data
