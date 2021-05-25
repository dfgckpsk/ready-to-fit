from .CVBase import CVBase, MlData
from sklearn.model_selection import KFold, StratifiedKFold
from tools.logging import logged


@logged
class SLCV(CVBase):

    def __init__(self, class_name, **kwargs):

        self.class_name = class_name
        self.n_splits = kwargs.get('n_splits', 5)
        if 'n_splits' in kwargs.keys():
            del kwargs['n_splits']

        self.one_n_splits = self.n_splits == 1
        if self.one_n_splits:
            self.n_splits = 2
        self.sl_cv_obj = None
        if self.class_name == 'kfold':
            self.sl_cv_obj = KFold(n_splits=self.n_splits, **kwargs)
        elif self.class_name == 'stratkfold':
            self.sl_cv_obj = StratifiedKFold(n_splits=self.n_splits, **kwargs)
        if self.sl_cv_obj is None:
            self._critical("Don't exist obj with name " + self.class_name)



    def split(self, mldata: MlData):

        indexes = mldata.get_indexes()
        labels = mldata.get_targets()

        if self.class_name == 'stratkfold':
            generator = self.sl_cv_obj.split(X=indexes, y=labels)
        else:
            generator = self.sl_cv_obj.split(X=indexes)
        values = ((indexes[train], indexes[test]) for train, test in generator)
        if self.one_n_splits:
            return [list(values)[-1]]
        print(values)
        return values
