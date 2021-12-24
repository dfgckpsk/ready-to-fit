from ..data.FeatureCreator import FeatureCreator, MlData
import pandas as pd
from ..tools.logging import logged


@logged
class RenameLabelsFC(FeatureCreator):

    def __init__(self, apply_feature=None, to_column=None, use_just_for_train=False, new_labels=None, new_label_names=None):
        FeatureCreator.__init__(self, apply_feature, to_column, use_just_for_train)
        self.new_labels = new_labels
        self.new_label_names = new_label_names

    def apply(self, ml_data: MlData):
        if self.new_labels is None:
            self._warning('leave labels is None')
            return ml_data
        if self.new_label_names is not None and len(self.new_label_names) != len(set(self.new_labels.values())):
            self._critical(f'new_label_names ({len(self.new_label_names)}) is not equal to new_labels '
                           f'({len(set(self.new_labels.values()))})')
            return ml_data
        labels = ml_data.get_targets()
        labels = pd.Series(labels, index=ml_data.get_indexes())
        if labels[~labels.isin(self.new_labels.keys())].shape[0] != 0:
            self._warning(f'Probably data has more unique labels than in new_labels. '
                          f'new_labels - {self.new_labels.keys()}')
            return ml_data

        labels = labels.apply(lambda x: self.new_labels.get(x))
        new_ml_data = ml_data.__copy__()
        new_ml_data.update_labels(labels)
        if self.new_label_names is not None:
            new_ml_data.label_names = self.new_label_names
        return new_ml_data
