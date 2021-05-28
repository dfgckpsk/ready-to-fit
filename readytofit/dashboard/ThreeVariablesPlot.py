import dash_core_components as dcc
import dash_html_components as html
import dash_table as dtable
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from readytofit.db.CreatedMlModels import CreatedMlModels
from typing import List, Tuple

DRAWDOWN_PARAM_1 = 'drawdown-param-1'
DRAWDOWN_PARAM_2 = 'drawdown-param-2'
DRAWDOWN_PARAM_3 = 'drawdown-param-3'
DRAWDOWN_PLOT_1 = 'drawdown-plot-1'
DRAWDOWN_PLOT_2 = 'drawdown-plot-2'
DRAWDOWN_PLOT_3 = 'drawdown-plot-3'
DRAWDOWN_RUN_ID = 'drawdown-run-id'
UPDATE_RUNS_ID = 'update-runs-id'


class ThreeVariablesPlot:

    def __init__(self):
        pass

    def layout(self):

        layout = []

        layout += [html.H2('ThreeVariablesPlot'),
                   html.Button('Update runs', id=UPDATE_RUNS_ID, n_clicks=0),
                    dcc.Loading(
                        id="loading-3",
                        type="circle",
                        children=html.Div(id="loading-output-3")
                    )]
        layout += self._run_id_drawdown_layout()
        layout += self._plot_parameters_layout()
        return layout

    def callback(self, app):

        self._callback_update_run_id(app)
        self._callback_update_plot_parameter_names(app)
        self._callback_select_experiment(app)

    def _run_id_drawdown_layout(self):
        return [html.H2('Run id'),
                dcc.Dropdown(id=DRAWDOWN_RUN_ID,
                             style={'width': '100%', 'display': 'inline-block'})]

    def _plot_parameters_layout(self):
        plot_parameters = ['line', 'dot']
        return [html.H2('parametersss'),
                dcc.Dropdown(id=DRAWDOWN_PARAM_1,
                             style={'width': '45%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PLOT_1,
                             options=[{'label': i, 'value': i} for i in plot_parameters],
                             style={'width': '45%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PARAM_2,
                             style={'width': '45%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PLOT_2,
                             options=[{'label': i, 'value': i} for i in plot_parameters],
                             style={'width': '45%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PARAM_3,
                             style={'width': '45%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PLOT_3,
                             options=[{'label': i, 'value': i} for i in plot_parameters],
                             style={'width': '45%', 'display': 'inline-block'})]

    def _callback_update_run_id(self, app):
        @app.callback(Output(DRAWDOWN_RUN_ID, 'options'),
                      Input(UPDATE_RUNS_ID, 'value'))
        def update_run_id(n_clicks):
            run_ids = self._get_runs_list()
            run_ids = [{'label': i, 'value': i} for i in run_ids*3]
            return run_ids

    def _callback_update_plot_parameter_names(self, app):
        @app.callback(Output(DRAWDOWN_PARAM_1, 'options'),
                      Output(DRAWDOWN_PARAM_2, 'options'),
                      Output(DRAWDOWN_PARAM_3, 'options'),
                      [Input(DRAWDOWN_RUN_ID, 'value')])
        def update_plot_parameter_names(run_id):
            paremeter_names = self._get_plot_parameter_names()
            paremeter_names = list(map(lambda x: [{'label': i, 'value': i} for i in x], paremeter_names))
            return paremeter_names

    def _callback_select_experiment(self, app):
        @app.callback(Output("loading-output-3", "children"),
                      Input(DRAWDOWN_PARAM_1, 'value'),
                      Input(DRAWDOWN_PARAM_2, 'value'),
                      Input(DRAWDOWN_PARAM_3, 'value'))
        def select_experiment(param_1, param_2, param_3):
            print('params', param_1, param_2, param_3)
            return True

    def _get_plot_parameter_names(self) -> Tuple[List[str], List[str], List[str]]:
        return (['param11', 'param12', 'param13'],
                ['param11', 'param12', 'param13'],
                ['param11', 'param12', 'param13'])

    def _get_runs_list(self) -> List[str]:
        return ['run_id1', 'run_id2', 'run_id3']