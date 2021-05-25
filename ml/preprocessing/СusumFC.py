from ml.data.FeatureCreator import FeatureCreator, MlData
from ml.data.MlDataFactory import MlDataFactory
import pandas as pd
import numpy as np
from tqdm import tqdm
from ml.tools.logging import logged


@logged
class CusumFC(FeatureCreator):

    def __init__(self, apply_feature=None, to_column=None, apply_after_label=False, use_just_for_train=False, thresold_feature=0):
        FeatureCreator.__init__(self, apply_feature, to_column, apply_after_label, use_just_for_train)
        self.thresold_feature = thresold_feature

    def apply(self, ml_data: MlData):

        symbols = pd.Series(ml_data.get_features(feature='symbol'),
                            index=ml_data.get_indexes())

        thresold_features = pd.DataFrame(ml_data.get_features(feature=self.thresold_feature),
                                      index=ml_data.get_indexes()).abs().min(axis=1)
        selected_indexes = pd.Series([0] * len(ml_data), index=ml_data.get_indexes())

        for symbol in list(set(symbols)):
            symbol_indexes = symbols[symbols == symbol].index

            close = ml_data.get_features(feature='close', indexes=symbol_indexes)
            close = pd.Series(close, index=symbol_indexes)

            sPos, sNeg = 0, 0

            diff = np.log(close.astype(float)).diff()
            for current_index in tqdm(symbol_indexes[1:]):
                try:
                    pos, neg = float(sPos + diff.loc[current_index]), float(sNeg + diff.loc[current_index])
                except Exception as e:
                    self._critical(f'Exception: {e}. index - {current_index}, diff - {diff.loc[current_index]}')
                    break
                current_thresold = thresold_features.loc[current_index]
                sPos, sNeg = max(0., pos), min(0., neg)
                if sNeg < -current_thresold:
                    sNeg = 0
                    selected_indexes.loc[current_index] = 1
                elif sPos > current_thresold:
                    sPos = 0
                    selected_indexes.loc[current_index] = 1

        selected_indexes = selected_indexes[selected_indexes == 1].index

        ml_data_ = MlDataFactory().ml_data_from_numpy(ml_data.get_features(indexes=selected_indexes),
                                                      ml_data.get_targets(indexes=selected_indexes),
                                                      indexes=selected_indexes,
                                                      label_names=ml_data.label_names,
                                                      feature_columns=ml_data.feature_names)

        return ml_data_
