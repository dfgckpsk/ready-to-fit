import unittest
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, StratifiedKFold
from ml.data.MlDataFactory import MlDataFactory
from ml.cv.SLCV import SLCV
from ml.cv.TimeSeriesSplit import TimeSeriesSplit
import numpy as np


class TestCV(unittest.TestCase):

    def setUp(self) -> None:
        data_sample = load_iris()
        self._ml_data = MlDataFactory().ml_data_from_numpy(data_sample.data,
                                                           data_sample.target)

    def test_slcv(self):

        for name, sklearn_cv in zip(['stratkfold', 'kfold'], [StratifiedKFold, KFold]):
            cv = SLCV(class_name=name)
            cv_sl = sklearn_cv()
            native_splitted = list(cv.split(self._ml_data))
            sklearn_splitted = list(cv_sl.split(self._ml_data.get_indexes(), self._ml_data.get_targets()))
            for native_ind, sklearn_ind in zip(native_splitted, sklearn_splitted):
                assert (native_ind[0] == sklearn_ind[0]).all()
                assert (native_ind[1] == sklearn_ind[1]).all()


    def test_timeseries(self):

        ml_data = MlDataFactory().ml_data_from_numpy(np.array(list(range(1, 25))).reshape(-1, 1),
                                                     np.array(list(range(1, 25))))
        cv = TimeSeriesSplit(n_splits=5)

        correct_split_train = [list(range(0, 4)),
                               list(range(4, 8)),
                               list(range(8, 12)),
                               list(range(12, 16)),
                               list(range(16, 20))]

        correct_split_test = [list(range(4, 8)),
                              list(range(8, 12)),
                              list(range(12, 16)),
                              list(range(16, 20)),
                              list(range(20, 24))]
        split_num = 0
        for train, test in cv.split(ml_data):
            assert correct_split_train[split_num] == list(train)
            assert correct_split_test[split_num] == list(test)
            split_num += 1


        cv = TimeSeriesSplit(n_splits=5, split_coeff=0.75)

        correct_split_train = [list(range(0, 6)),
                               list(range(4, 10)),
                               list(range(8, 14)),
                               list(range(12, 18)),
                               list(range(16, 22))]

        correct_split_test = [list(range(6, 8)),
                              list(range(10, 12)),
                              list(range(14, 16)),
                              list(range(18, 20)),
                              list(range(22, 24))]
        split_num = 0
        for train, test in cv.split(ml_data):
            assert correct_split_train[split_num] == list(train)
            assert correct_split_test[split_num] == list(test)
            split_num += 1


        cv = TimeSeriesSplit(n_splits=5, mode='wide')

        correct_split_train = [list(range(0, 4)),
                               list(range(0, 8)),
                               list(range(0, 12)),
                               list(range(0, 16)),
                               list(range(0, 20))]

        correct_split_test = [list(range(4, 24)),
                              list(range(8, 24)),
                              list(range(12, 24)),
                              list(range(16, 24)),
                              list(range(20, 24))]
        split_num = 0
        for train, test in cv.split(ml_data):
            assert correct_split_train[split_num] == list(train)
            assert correct_split_test[split_num] == list(test)
            split_num += 1


        cv = TimeSeriesSplit(n_splits=1, mode='normal')

        correct_split_train = [list(range(0, 12))]

        correct_split_test = [list(range(12, 24))]
        split_num = 0
        for train, test in cv.split(ml_data):
            print(train, test)
            assert correct_split_train[split_num] == list(train)
            assert correct_split_test[split_num] == list(test)


        cv = TimeSeriesSplit(n_splits=1, mode='normal')

        correct_split_train = [list(range(0, 12))]

        correct_split_test = [list(range(12, 24))]
        split_num = 0
        for train, test in cv.split(ml_data):
            assert correct_split_train[split_num] == list(train)
            assert correct_split_test[split_num] == list(test)


        cv = TimeSeriesSplit(n_splits=1, mode='normal', split_coeff=0.75)

        correct_split_train = [list(range(0, 18))]

        correct_split_test = [list(range(18, 24))]
        split_num = 0
        for train, test in cv.split(ml_data):
            assert correct_split_train[split_num] == list(train)
            assert correct_split_test[split_num] == list(test)
