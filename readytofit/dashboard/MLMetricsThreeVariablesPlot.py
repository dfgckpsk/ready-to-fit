from .ThreeVariablesPlot import ThreeVariablesPlot, go, dash, List, CHECKLIST_OPTIONS
from readytofit.dashboard.MLDashboardData import MlDashboardData
import datetime
import time


class MLMetricsThreeVariablesPlot(ThreeVariablesPlot):

    def __init__(self, app: dash.Dash, ml_dashboard_manager: MlDashboardData):
        ThreeVariablesPlot.__init__(self, app)
        self.ml_dashboard_manager = ml_dashboard_manager
        self.ts_metrics = []
        self.prev_data_id = None
        self.already_updated = False

    def _get_points(self, param_name, plot_type, secondary=False) -> go.Scatter:
        if len(self.ts_metrics) == 0:
            return
        ts_metric = list(filter(lambda x: x.value_name == param_name, self.ts_metrics))[0]
        if plot_type == 'markers':
            return go.Scatter(x=ts_metric.indexes,
                              y=ts_metric.float_values,
                              mode='markers',
                              name=param_name,
                              yaxis='y1' if secondary else 'y2')

        return go.Scatter(x=ts_metric.indexes,
                          y=ts_metric.float_values,
                          name=param_name,
                          yaxis='y1' if secondary else 'y2')

    def _update_figure(self, param_1, param_2, param_3, plot_type_1, plot_type_2, plot_type_3, data_id, selected_secondary) -> go.Figure:
        self._update_ts_metrics(data_id)
        data = []
        for param, plot_type, sec_option_names in zip([param_1, param_2, param_3],
                                                      [plot_type_1, plot_type_2, plot_type_3],
                                                      CHECKLIST_OPTIONS):
            if param is not None and plot_type is not None:
                points = self._get_points(param, plot_type, selected_secondary is not None and sec_option_names in selected_secondary)
                if points is not None:
                    data.append(points)
        layout = go.Layout(yaxis=dict(),
                           yaxis2=dict(overlaying='y',
                                       side='right'))
        return go.Figure(data=data, layout=layout)

    def _get_parameter_names_for_plot(self, data_id: str) -> List[str]:
        try:
            self.data_id = data_id
            self._update_ts_metrics(data_id)

            return list(map(lambda x: x.value_name, self.ts_metrics))
        except BaseException as e:
            print(e)
            # self.ts_metrics = None
            return []

    def _update_ts_metrics(self, data_id):
        if data_id != self.prev_data_id and not self.already_updated:
            self.ts_metrics = None
            self.already_updated = True
            self.prev_data_id = data_id
            split_num = None
            if '_' in data_id:
                data_id, split_num = data_id.split('_')
            data_id = datetime.datetime.strptime(data_id, '%Y-%m-%d %H:%M:%S.%f')
            self.ts_metrics = self.ml_dashboard_manager.get_ts_metrics(data_id)
            if split_num is not None:
                self.ts_metrics = list(filter(lambda x: x.split_num == int(split_num), self.ts_metrics))
            self.already_updated = False
        while self.already_updated:
            time.sleep(0.5)

    def _get_data_id_list(self) -> List[str]:
        import time
        time.sleep(1)
        run_ids = self.ml_dashboard_manager.get_run_id_by_exp()
        run_ids = list(map(lambda x: (x[0].replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S.%f'), x[1]), run_ids))
        run_ids = list(map(lambda x: x[0] + '_' + str(x[1]) if x[1] is not None else x[0], run_ids))
        run_ids = sorted(run_ids, reverse=True)

        return run_ids
