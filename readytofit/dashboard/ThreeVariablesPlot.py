import dash_core_components as dcc
import dash_html_components as html
import dash_table as dtable
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from readytofit.db.CreatedMlModels import CreatedMlModels
from typing import List, Tuple
from readytofit.dashboard.base.DashboardObject import DashboardObject, dash
import plotly.graph_objs as go

DRAWDOWN_PARAM_1 = 'drawdown-param-1'
DRAWDOWN_PARAM_2 = 'drawdown-param-2'
DRAWDOWN_PARAM_3 = 'drawdown-param-3'
DRAWDOWN_PLOT_1 = 'drawdown-plot-1'
DRAWDOWN_PLOT_2 = 'drawdown-plot-2'
DRAWDOWN_PLOT_3 = 'drawdown-plot-3'
DRAWDOWN_RUN_ID = 'drawdown-run-id'
UPDATE_RUNS_ID = 'update-runs-id'
FIGURE_ID = 'figure-id'


class ThreeVariablesPlot(DashboardObject):

    def __init__(self, app: dash.Dash):
        DashboardObject.__init__(self, app)
        self.callback_objects = [self._callback_update_data_id,
                                 self._callback_update_plot_parameter_names,
                                 self._callback_select_data_options]

    def layout(self):

        layout = []

        layout += [html.H2('ThreeVariablesPlot'),
                   html.Button('Update runs', id=UPDATE_RUNS_ID, n_clicks=0),
                    dcc.Loading(
                        id="loading-3",
                        type="circle",
                        children=html.Div(id="loading-output-3")
                    )]
        parameters_layouts = self._layout_data_id_drawdown()
        parameters_layouts += self._layout_plot_parameters()
        figure_layout = self._layout_figure()
        data_layout = html.Div([html.Div(figure_layout, style={'width': '70%', 'display': 'inline-block'}),
                                html.Div(parameters_layouts, style={'width': '30%', 'display': 'inline-block'})],
                               style={'width': '100%', 'display': 'inline-block'})
        return [html.Div(layout), data_layout]

    def _layout_data_id_drawdown(self):
        return [html.H2('Run id'),
                dcc.Dropdown(id=DRAWDOWN_RUN_ID,
                             style={'width': '90%', 'display': 'inline-block'})]

    def _layout_plot_parameters(self):
        plot_parameters = ['line', 'markers']
        return [html.H2('parametersss'),
                dcc.Dropdown(id=DRAWDOWN_PARAM_1,
                             style={'width': '50%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PLOT_1,
                             options=[{'label': i, 'value': i} for i in plot_parameters],
                             value=plot_parameters[0],
                             style={'width': '50%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PARAM_2,
                             style={'width': '50%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PLOT_2,
                             options=[{'label': i, 'value': i} for i in plot_parameters],
                             value=plot_parameters[0],
                             style={'width': '50%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PARAM_3,
                             style={'width': '50%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PLOT_3,
                             options=[{'label': i, 'value': i} for i in plot_parameters],
                             value=plot_parameters[0],
                             style={'width': '50%', 'display': 'inline-block'})]

    def _layout_figure(self):
        return [dcc.Graph(id=FIGURE_ID)]

    def _callback_update_data_id(self, app):
        @app.callback(Output(DRAWDOWN_RUN_ID, 'options'),
                      Input(UPDATE_RUNS_ID, 'value'))
        def update_run_id(n_clicks):
            run_ids = self._get_data_id_list()
            run_ids = [{'label': i, 'value': i} for i in run_ids*3]
            return run_ids

    def _callback_update_plot_parameter_names(self, app):
        @app.callback(Output(DRAWDOWN_PARAM_1, 'options'),
                      Output(DRAWDOWN_PARAM_2, 'options'),
                      Output(DRAWDOWN_PARAM_3, 'options'),
                      [Input(DRAWDOWN_RUN_ID, 'value')])
        def update_plot_parameter_names(data_id):
            paremeter_names = self._get_parameter_names_for_plot(data_id)
            paremeter_names = [{'label': i, 'value': i} for i in paremeter_names]
            return paremeter_names, paremeter_names, paremeter_names

    def _callback_select_data_options(self, app):
        @app.callback(Output('loading-output-3', 'children'),
                      Output(FIGURE_ID, 'figure'),
                      Input(DRAWDOWN_PARAM_1, 'value'),
                      Input(DRAWDOWN_PARAM_2, 'value'),
                      Input(DRAWDOWN_PARAM_3, 'value'),
                      Input(DRAWDOWN_PLOT_1, 'value'),
                      Input(DRAWDOWN_PLOT_2, 'value'),
                      Input(DRAWDOWN_PLOT_3, 'value'))
        def select_experiment(param_1, param_2, param_3, plot_type_1, plot_type_2, plot_type_3):
            figure = self._update_figure(param_1,
                                         param_2,
                                         param_3,
                                         plot_type_1,
                                         plot_type_2,
                                         plot_type_3)
            if figure is None:
                raise PreventUpdate
            return True, figure

    def _get_parameter_names_for_plot(self, data_id: str) -> List[str]:
        return ['param11', 'param12', 'param13']

    def _get_data_id_list(self) -> List[str]:
        return ['run_id1', 'run_id2', 'run_id3']

    def _update_figure(self, param_1, param_2, param_3, plot_type_1, plot_type_2, plot_type_3) -> go.Figure:
        # return go.Figure()
        return None
