from ml.data.FeatureCreator import FeatureCreator, MlData
import pandas as pd
import numpy as np
from ml.tools.logging import logged
import datetime


@logged
class VolatilityFC(FeatureCreator):

    def __init__(self,
                 apply_feature=None,
                 to_column=None,
                 apply_after_label=False,
                 use_just_for_train=False,
                 values_to_rolling=None,
                 minutes_to_calculate_return=None):
        FeatureCreator.__init__(self, apply_feature, to_column, apply_after_label, use_just_for_train)
        self.values_to_rolling = values_to_rolling
        self.minutes_to_calculate_return = minutes_to_calculate_return
        if self.values_to_rolling is None:
            self._warning('values_to_rolling is None')
        if self.minutes_to_calculate_return is None:
            self._warning('minutes_to_calculate_return is None')

    def apply(self, ml_data: MlData):

        feature_name = f'volatility_{self.minutes_to_calculate_return}_{self.values_to_rolling}'

        symbols = pd.Series(ml_data.get_features(feature='symbol'),
                            index=ml_data.get_indexes())
        volatility = pd.Series([np.nan]*len(ml_data),
                               index=ml_data.get_indexes())
        for symbol in list(set(symbols)):
            symbol_indexes = symbols[symbols == symbol].index

            close = ml_data.get_features(feature='close', indexes=symbol_indexes)
            close = pd.Series(close, index=list(map(lambda x: datetime.datetime.utcfromtimestamp(x/1e9),
                                                    ml_data.get_features(feature='timestamp', indexes=symbol_indexes))))

            prev_close_indexes = close.index.searchsorted(close.index - pd.Timedelta(minutes=1))
            prev_close_indexes = prev_close_indexes[prev_close_indexes > 0]
            prev_close_indexes = pd.Series(close.index[prev_close_indexes - 1],
                                           index=close.index[close.shape[0] - prev_close_indexes.shape[0]:])
            try:
                returns = close.loc[prev_close_indexes.index] / close.loc[prev_close_indexes.values].values - 1
            except Exception as e:
                self._critical(f'error: {e}\nplease confirm no duplicate indices')

            volatility_symbol = pd.Series(index=close.index)
            volatility_symbol.loc[returns.index] = returns.ewm(span=self.values_to_rolling).std().rename(feature_name)
            volatility.loc[symbol_indexes] = volatility_symbol.values


        ml_data.update_features(volatility, feature_name)

        return ml_data
