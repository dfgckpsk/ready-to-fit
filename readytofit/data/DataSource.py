from readytofit.db.MLDatabaseManager import MLDatabaseManager
from readytofit.db.MlParameterFactory import MLParameterFactory, MlParameterType
from readytofit.data.MlDataFactory import MlDataFactory, MlData
from datetime import datetime


class DataSource:

    def __init__(self, run_id: datetime, parameters: dict = {}, ml_database_manager: MLDatabaseManager = None):
        self.ml_database_manager = ml_database_manager
        self.run_id = run_id
        self.parameters = parameters
        self.output = None

    def _generate_data(self) -> MlData:
        pass

    def generate_data(self) -> MlData:
        if self.output is None:
            self.output = self._generate_data()
            data_info = {}
            data_info['features_count'] = len(self.output.feature_names)
            data_info['length'] = len(self.output)
            data_info['weights_here'] = int(self.output is not None)
            self.log_parameters({**self.parameters, **data_info})
        return self.output

    def log_parameters(self, parameters: dict):
        if self.ml_database_manager is not None:
            ml_parameters = MLParameterFactory.from_dict(parameters, self.run_id, MlParameterType.Data)
            self.ml_database_manager.insert_parameters(ml_parameters, immediate_insert=True)
