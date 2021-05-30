from readytofit.db.BaseDatabase import BaseDatabase
from readytofit.db.CreatedMlModels import CreatedMlModels
from readytofit.db.ExperimentInfo import ExperimentInfo
from readytofit.db.MlTsValues import MlTsValues
from readytofit.db.MlParameter import MlParameter
from datetime import datetime
from typing import List


class MLDatabaseManager:

    def __init__(self, database: BaseDatabase):
        self.db = database

    def get_run_result(self, run_id: datetime):

        query = f"""
        select * 
        from backtest.created_ml_models 
        where run_id_full == '{run_id.strftime('%Y-%m-%d %H:%M:%S.%f')}';
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

    def get_run_id_by_exp(self, exp_id: str = None):

        exp_str = f"where exp_id == '{exp_id}'" if exp_id is not None else ''
        query = f"""
        select run_id_full
        from backtest.created_ml_models 
        {exp_str}
        group by run_id_full;
        """

        results = self.db.get_all(query)
        return results

    def get_exp_ids(self) -> List[ExperimentInfo]:

        query = f"""
        select exp_id,
               count(distinct (run_id_full)) as uniq_count_runs,
               count(run_id_full) as count_runs,
               min(run_id_full) as min_date,
               max(run_id_full) as max_date,
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

    def get_ml_parameters(self, run_id: datetime) -> List[MlParameter]:

        query = f""" 
                    select * 
                    from backtest.ml_parameters 
                    where run_id == '{run_id}'
                    """
        results = self.db.get_all(query)
        results = list(map(lambda x: MlParameter(*x), results))
        return results

    def insert_parameters(self, ml_paremeters: List[MlParameter], immediate_insert=True):
        self.db.insert_row(ml_paremeters, table='ml_parameters', immediate_insert=immediate_insert)

    def insert_experiment(self, created_ml_model: CreatedMlModels, immediate_insert=True):
        self.db.insert_row([created_ml_model], table='created_ml_models', immediate_insert=immediate_insert)

    def insert_ts_values(self, ml_ts_values: MlTsValues, immediate_insert=True):
        self.db.insert_row([ml_ts_values], table='ml_ts_values', immediate_insert=immediate_insert)
