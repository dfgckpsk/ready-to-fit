import numpy as np
from .CVBase import CVBase, MlData
from readytofit.tools.logging import logged


WIDE_MODE = 'wide'
LEFT_MODE = 'left'
RIGHT_MODE = 'right'
NORMAL_MODE = 'normal'


@logged
class TimeSeriesSplit(CVBase):

    """
        If ind - [1,2,3,4,5,6,7,8,9,10,11,12,13], n_splits=5

        wide - last_fold_include=False
        [1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        [1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12, 13]
        [1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]
        [1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13]
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13]

        left - last_fold_include=False
        [1, 2, 3], [4, 5]
        [1, 2, 3, 4, 5], [6, 7]
        [1, 2, 3, 4, 5, 6, 7], [8, 9]
        [1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11]
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13]

        left - last_fold_include=True
        [1, 2, 3], [4, 5, 6]
        [1, 2, 3, 4, 5, 6], [7, 8, 9]
        [1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11]
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13]
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [12, 13]

        right - last_fold_include=False
        [1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        [4, 5], [6, 7, 8, 9, 10, 11, 12, 13]
        [6, 7], [8, 9, 10, 11, 12, 13]
        [8, 9], [10, 11, 12, 13]
        [10, 11], [12, 13]

        normal - last_fold_include=False
        [1, 2, 3], [4, 5]
        [4, 5], [6, 7]
        [6, 7], [8, 9]
        [8, 9], [10, 11]
        [10, 11], [12, 13]

    """

    def __init__(self, **kwargs):

        """
        last_fold_include - if True return last values
        :param kwargs: mode - wide, left, right, normal
        """
        self.n_splits = kwargs.get('n_splits', 3)
        self.mode = kwargs.get('mode', 'normal')
        self.last_fold_include = kwargs.get('last_fold_include', True)
        self.split_coeff = kwargs.get('split_coeff')
        if self.split_coeff is not None and self.mode != NORMAL_MODE:
            self._warning('split_coeff supports just in "normal" mode')
        if self.split_coeff is not None and not (0 <= self.split_coeff <= 1):
            self._critical('Split coeffs should be between 0 and 1, e.g. 0.6')

    def split(self, mldata: MlData):

        """
        Split generator
        :param mldata:
        :return:
        """

        max_split = self.n_splits + 1 if self.last_fold_include else self.n_splits
        split_ind = np.array_split(mldata.get_indexes(), max_split)

        data_len = len(mldata)
        indexes = mldata.get_indexes()

        for n in range(self.n_splits):
            last_n = min(n + 1, n+1)

            if self.mode == NORMAL_MODE:
                train_start_ind = data_len // (self.n_splits + 1) * n
                test_end_ind = data_len // (self.n_splits + 1) * (n+2)
                if self.n_splits > 1:
                    train_end_ind = data_len // (self.n_splits + 1) * (n+1) if self.split_coeff is None \
                        else int(train_start_ind + (test_end_ind - train_start_ind) * self.split_coeff)
                else:
                    train_end_ind = data_len // 2 if self.split_coeff is None else int(data_len * self.split_coeff)
                yield indexes[train_start_ind: train_end_ind], indexes[train_end_ind: test_end_ind]
            elif self.mode == WIDE_MODE:
                yield np.concatenate(split_ind[0:n+1]), np.concatenate(split_ind[last_n:]) if last_n < max_split else []
            elif self.mode == LEFT_MODE:
                yield np.concatenate(split_ind[0:n+1]), split_ind[last_n] if last_n < max_split else []
            elif self.mode == RIGHT_MODE:
                yield split_ind[n], np.concatenate(split_ind[last_n:]) if last_n < max_split else []
