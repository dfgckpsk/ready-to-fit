from ml.db.MLDatabaseManager import MLDatabaseManager
from datetime import datetime


class MlDashboardManager:

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

    def get_ts_metrics(self, run_id: datetime):
        ts_metrics = self.database_manager.get_ts_metrics(run_id)
        metrics_names = list(map(lambda x: x.value_name, ts_metrics))

    def get_run_id_by_exp(self, exp_id: str):
        return self.database_manager.get_run_id_by_exp(exp_id)
