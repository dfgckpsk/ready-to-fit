from ml.db.BaseDatabase import BaseDatabase
from ml.db.CreatedMlModels import CreatedMlModels
from ml.db.ExperimentInfo import ExperimentInfo
from ml.db.MlTsValues import MlTsValues
from datetime import datetime
from typing import List


class MLDatabaseManager:

    def __init__(self, database: BaseDatabase):
        self.db = database

    def get_run_result(self, run_id: datetime):

        query = f"""
        select * 
        from backtest.created_ml_models 
        where run_id == '{run_id.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}';
        """

        results = self.db.get_all(query)
        return results

    def get_experiment_results(self, exp_id: str) -> List[CreatedMlModels]:

        query = f"""
        select * 
        from backtest.created_ml_models 
        where exp_id == '{exp_id}';
        """

        results = self.db.get_all(query)
        results = list(map(lambda x: CreatedMlModels(*x), results))
        return results

    def get_run_id_by_exp(self, exp_id: str):

        query = f"""
        select run_id
        from backtest.created_ml_models 
        where exp_id == '{exp_id}'
        group by run_id;
        """

        results = self.db.get_all(query)
        return results

    def get_exp_ids(self) -> List[ExperimentInfo]:

        query = f"""
        select exp_id,
               count(distinct (run_id)) as uniq_count_runs,
               count(run_id) as count_runs,
               min(run_id) as min_date,
               max(run_id) as max_date,
               max(split_num) as max_split
        from backtest.created_ml_models
        group by exp_id;
        """

        results = self.db.get_all(query)
        results = list(map(lambda x: ExperimentInfo(*x), results))
        return results

    def get_ts_metrics(self, run_id: datetime, value_name: str = None) -> List[MlTsValues]:

        query = f""" 
                    select * 
                    from backtest.ml_ts_values 
                    where run_id == '{run_id}'
                    """
        if value_name is not None:
            query += f"and value_name == '{value_name}'"
        results = self.db.get_all(query)
        results = list(map(lambda x: MlTsValues(*x), results))
        return results

    def insert_experiment(self, created_ml_model: CreatedMlModels, immediate_insert=True):
        self.db.insert_row([created_ml_model], table='created_ml_models', immediate_insert=immediate_insert)

    def insert_ts_values(self, ml_ts_values: MlTsValues, immediate_insert=True):
        self.db.insert_row([ml_ts_values], table='ml_ts_values', immediate_insert=immediate_insert)
