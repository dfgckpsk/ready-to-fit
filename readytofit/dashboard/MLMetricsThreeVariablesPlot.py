from .FiveVariablesPlot import ThreeVariablesPlot, go, dash, List
from readytofit.dashboard.MLDashboardData import MlDashboardData
import datetime


class MLMetricsThreeVariablesPlot(ThreeVariablesPlot):

    def __init__(self, app: dash.Dash, ml_dashboard_manager: MlDashboardData):
        ThreeVariablesPlot.__init__(self, app)
        self.ml_dashboard_manager = ml_dashboard_manager

    def _get_points(self, param_name, plot_type, secondary=False) -> go.Scatter:
        if len(self.data) == 0:
            return
        data = list(filter(lambda x: x.value_name == param_name, self.data))[0]
        if plot_type == 'markers':
            return go.Scatter(x=data.indexes,
                              y=data.float_values,
                              mode='markers',
                              name=param_name,
                              yaxis='y1' if secondary else 'y2')

        return go.Scatter(x=data.indexes,
                          y=data.float_values,
                          name=param_name,
                          yaxis='y1' if secondary else 'y2')

    def _get_parameter_names_for_plot(self, data_id: str) -> List[str]:
        try:
            self.data_id = data_id
            self._update_data(data_id)

            return list(map(lambda x: x.value_name, self.data))
        except BaseException as e:
            print(e)
            return []

    def _update_data_callback(self, data_id):
        if data_id is None:
            return
        split_num = None
        if '_' in data_id:
            data_id, split_num = data_id.split('_')
        data_id = datetime.datetime.strptime(data_id, '%Y-%m-%d %H:%M:%S.%f')
        self.data = self.ml_dashboard_manager.get_ts_metrics(data_id)
        if split_num is not None:
            self.data = list(filter(lambda x: x.split_num == int(split_num), self.data))

    def _get_data_id_list(self) -> List[str]:
        import time
        time.sleep(1)
        run_ids = self.ml_dashboard_manager.get_run_id_in_ts_metrics()
        run_ids = list(map(lambda x: (x[0].replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S.%f'), x[1]), run_ids))
        run_ids = list(map(lambda x: x[0] + '_' + str(x[1]) if x[1] is not None else x[0], run_ids))
        run_ids = sorted(run_ids, reverse=True)

        return run_ids
