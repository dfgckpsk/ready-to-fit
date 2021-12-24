from ..db.MLDatabaseManager import MLDatabaseManager, List, MlTsValues
from datetime import datetime


class MlDashboardData:

    def __init__(self, database_manager: MLDatabaseManager):
        self.database_manager = database_manager
        self.prev_experiment_output = None
        self.prev_experiment_id = None

    def get_exp_ids(self):
        return self.database_manager.get_exp_ids()

    def get_experiment_results(self, exp_id: str):
        if self.prev_experiment_id is None or self.prev_experiment_output is None or self.prev_experiment_id != exp_id:
            experiments = self.database_manager.get_experiment_results(exp_id)
            self.prev_experiment_id = exp_id
            self.prev_experiment_output = experiments
        return self.prev_experiment_output

    def get_ts_metrics(self, run_id: datetime) -> List[MlTsValues]:
        ts_metrics = self.database_manager.get_ts_metrics(run_id)
        return ts_metrics

    def get_run_id_by_exp(self, exp_id: str = None):
        return self.database_manager.get_run_id_by_exp(exp_id)

    def get_run_id_in_ts_metrics(self):
        return self.database_manager.get_run_id_in_ts_metrics()
