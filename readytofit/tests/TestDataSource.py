from readytofit.data.DataSource import DataSource, MlData, MlDataFactory
from sklearn.datasets import load_iris


class TestDataSource(DataSource):

    def _generate_data(self) -> MlData:

        data_sample = load_iris()
        ml_data = MlDataFactory().ml_data_from_numpy(data_sample.data,
                                                           data_sample.target)
        return ml_data
