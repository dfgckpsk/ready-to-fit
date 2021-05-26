from .ThreeVariablesPlot import ThreeVariablesPlot, List, html, dcc, DRAWDOWN_RUN_ID, UPDATE_RUNS_ID, Input, Output, Tuple
from ml.dashboard.MLDashboardManager import MlDashboardManager

DRAWDOWN_EXP_ID = 'drawdown-exp-id'


class MLMetricsThreeVariablesPlot(ThreeVariablesPlot):

    def __init__(self, ml_dashboard_manager: MlDashboardManager):
        ThreeVariablesPlot.__init__(self)
        self.ml_dashboard_manager = ml_dashboard_manager

    def _run_id_drawdown_layout(self):
        layout = html.Div([html.Div([html.H2('Exp id'),
                                     dcc.Dropdown(id=DRAWDOWN_EXP_ID, style={'width': '100%', 'display': 'inline-block'})],
                                    style={'width': '30%', 'display': 'inline-block'}),
                           html.Div(ThreeVariablesPlot._run_id_drawdown_layout(self),
                                    style={'width': '30%', 'display': 'inline-block'})],
                          style={'width': '100%', 'display': 'inline-block'})
        return [layout]

    def callback(self, app):
        ThreeVariablesPlot.callback(self, app)
        self._callback_update_exp_id(app)

    def _callback_update_exp_id(self, app):
        @app.callback(Output(DRAWDOWN_EXP_ID, 'options'),
                      Input(UPDATE_RUNS_ID, 'value'))
        def update_exp_id(n_clicks):
            import time
            time.sleep(1)
            exp_ids = self._get_exp_list()
            exp_ids = [{'label': i, 'value': i} for i in exp_ids]
            return exp_ids

    def _callback_update_run_id(self, app):
        @app.callback(Output(DRAWDOWN_RUN_ID, 'options'),
                      Input(DRAWDOWN_EXP_ID, 'value'))
        def update_run_id(exp_id):
            run_ids = self._get_runs_list(exp_id)
            run_ids = [{'label': i, 'value': i} for i in run_ids]
            return run_ids

    def _get_runs_list(self, exp_id) -> List[str]:
        if exp_id is not None:
            run_ids = self.ml_dashboard_manager.get_run_id_by_exp(exp_id)
            run_ids = list(map(lambda x: x[0], run_ids))
            run_ids = sorted(run_ids)
            return run_ids
        return []

    def _get_exp_list(self) -> List[str]:
        experiment_infos = self.ml_dashboard_manager.get_exp_ids()
        experiments_ids = list(map(lambda x: x.exp_id, experiment_infos))
        return experiments_ids
