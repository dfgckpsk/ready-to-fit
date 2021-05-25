from unittest import TestCase
from db.MLDatabaseManager import MLDatabaseManager, MlTsValues, CreatedMlModels
from db.clickhouse import Clickhouse
import datetime
import time


class TestMLDatabaseManager(TestCase):

    def setUp(self) -> None:
        self._database = Clickhouse.init_database('100.126.106.82:9901:code:SIa/PNNW:backtest')

    def test_exp_id_getting(self):
        ml_model_manager = MLDatabaseManager(self._database)
        exp_count = 12
        exp_id = 'test_exp_id_getting'
        for i in range(exp_count):
            created_ml_models = CreatedMlModels(exp_id=exp_id,
                                                run_id=datetime.datetime.now(),
                                                current_index=datetime.datetime.now(),
                                                config='{}',
                                                model_id='model_id_test',
                                                seed=10)
            ml_model_manager.insert_experiment(created_ml_models)

        experiements = ml_model_manager.get_experiment_results(exp_id)
        for exp in experiements:
            assert exp.exp_id == exp_id
        assert len(experiements) == exp_count

        experiements_unique = ml_model_manager.get_exp_ids()
        experiements_unique = list(filter(lambda x: x.exp_id == exp_id, experiements_unique))[0]

        assert experiements_unique.uniq_count_runs == exp_count
        assert experiements_unique.count_runs == exp_count
        assert experiements_unique.min_date == min(list(map(lambda x: x.run_id, experiements)))
        assert experiements_unique.max_date == max(list(map(lambda x: x.run_id, experiements)))

        self._database.execute(f"ALTER TABLE backtest.created_ml_models DELETE WHERE exp_id == '{exp_id}'")
        time.sleep(1)

        experiements = ml_model_manager.get_experiment_results(exp_id)
        assert len(experiements) == 0

    def test_ts_model_metrics(self):
        ml_model_manager = MLDatabaseManager(self._database)

        dt = datetime.datetime.now() - datetime.timedelta(days=2000)
        metric_name = 'metric_name_test'
        float_values = [1.0, 2.0, 3.0]
        ml_ts_values = MlTsValues(run_id=dt,
                                  value_name=metric_name,
                                  float_values=float_values)
        ml_model_manager.insert_ts_values(ml_ts_values)

        for metric in [metric_name, None]:
            got_ml_ts_values = ml_model_manager.get_ts_metrics(dt, metric)[0]

            assert ml_ts_values.run_id == got_ml_ts_values.run_id
            assert ml_ts_values.value_name == got_ml_ts_values.value_name and got_ml_ts_values.value_name == metric_name
            assert ml_ts_values.float_values == got_ml_ts_values.float_values
            assert ml_ts_values.int_values == got_ml_ts_values.int_values
            assert ml_ts_values.split_num == got_ml_ts_values.split_num

        self._database.execute(f"ALTER TABLE backtest.ml_ts_values DELETE WHERE run_id == '{dt}'")
