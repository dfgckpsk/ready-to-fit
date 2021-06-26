import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from typing import List
from readytofit.dashboard.base.DashboardObject import DashboardObject, dash
import plotly.graph_objs as go
import time

DRAWDOWN_PARAM_1 = 'drawdown-param-1'
DRAWDOWN_PARAM_2 = 'drawdown-param-2'
DRAWDOWN_PARAM_3 = 'drawdown-param-3'
DRAWDOWN_PARAM_4 = 'drawdown-param-4'
DRAWDOWN_PARAM_5 = 'drawdown-param-5'
DRAWDOWN_PLOT_1 = 'drawdown-plot-1'
DRAWDOWN_PLOT_2 = 'drawdown-plot-2'
DRAWDOWN_PLOT_3 = 'drawdown-plot-3'
DRAWDOWN_PLOT_4 = 'drawdown-plot-4'
DRAWDOWN_PLOT_5 = 'drawdown-plot-5'
DRAWDOWN_RUN_ID = 'drawdown-run-id'
SECONDARY_CHECKLIST_ID = 'secondary-checklist-id'
UPDATE_RUNS_ID = 'update-runs-id'
FIGURE_ID = 'figure-id'
CHECKLIST_OPTIONS = ['1 secondary',
                     '2 secondary',
                     '3 secondary',
                     '4 secondary',
                     '5 secondary']


class FiveVariablesPlot(DashboardObject):

    def __init__(self, app: dash.Dash, title: str = 'ThreeVariablesPlot'):
        DashboardObject.__init__(self, app)
        self.callback_objects = [self._callback_update_data_id,
                                 self._callback_update_plot_parameter_names,
                                 self._callback_select_data_options]
        self.data_id = None
        self.data = []
        self.prev_data_id = None
        self.data_already_updated = False
        self.title = title

    def layout(self):

        layout = []

        layout += [html.H2(self.title),
                   html.Button('Update runs', id=UPDATE_RUNS_ID, n_clicks=0)]
        parameters_layouts = self._layout_data_id_drawdown()
        parameters_layouts += self._layout_plot_parameters()
        figure_layout = self._layout_figure()
        data_layout = html.Div([html.Div(figure_layout, style={'width': '80%', 'display': 'inline-block'}),
                                html.Div(parameters_layouts, style={'width': '20%',
                                                                    'display': 'inline-block',
                                                                    'marginBottom': 100})],
                               style={'width': '100%', 'display': 'inline-block'})
        return [html.Div(layout), data_layout]

    def _layout_data_id_drawdown(self):
        return [html.H2('Run id'),
                dcc.Loading(
                    id="loading-4",
                    type="circle",
                    children=
                    dcc.Dropdown(id=DRAWDOWN_RUN_ID,
                             style={'width': '90%', 'display': 'inline-block'}))]

    def _layout_plot_parameters(self):
        plot_parameters = ['line', 'markers']
        drawdowns_layout = [dcc.Dropdown(id=DRAWDOWN_PARAM_1,
                             style={'width': '90%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PLOT_1,
                             options=[{'label': i, 'value': i} for i in plot_parameters],
                             value=plot_parameters[0],
                             style={'width': '90%', 'display': 'inline-block'}),

                dcc.Dropdown(id=DRAWDOWN_PARAM_2,
                             style={'width': '90%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PLOT_2,
                             options=[{'label': i, 'value': i} for i in plot_parameters],
                             value=plot_parameters[0],
                             style={'width': '90%', 'display': 'inline-block'}),

                dcc.Dropdown(id=DRAWDOWN_PARAM_3,
                             style={'width': '90%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PLOT_3,
                             options=[{'label': i, 'value': i} for i in plot_parameters],
                             value=plot_parameters[0],
                             style={'width': '90%', 'display': 'inline-block'}),

                dcc.Dropdown(id=DRAWDOWN_PARAM_4,
                             style={'width': '90%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PLOT_4,
                             options=[{'label': i, 'value': i} for i in plot_parameters],
                             value=plot_parameters[0],
                             style={'width': '90%', 'display': 'inline-block'}),

                dcc.Dropdown(id=DRAWDOWN_PARAM_5,
                             style={'width': '90%', 'display': 'inline-block'}),
                dcc.Dropdown(id=DRAWDOWN_PLOT_5,
                             options=[{'label': i, 'value': i} for i in plot_parameters],
                             value=plot_parameters[0],
                             style={'width': '90%', 'display': 'inline-block'}),

                dcc.Checklist(id=SECONDARY_CHECKLIST_ID,
                              options=[{'label': i, 'value': i} for i in CHECKLIST_OPTIONS],
                              labelStyle={'width': '90%', 'display': 'inline-block'})
                ]
        return [html.H4('settings'),
                dcc.Loading(
                    id="loading-5",
                    type="circle",
                    children=html.Div(drawdowns_layout))]

    def _layout_figure(self):
        return [dcc.Loading(
                        id="loading-3",
                        type="circle",
                        children=dcc.Graph(id=FIGURE_ID, style={'height': '90vh'}))]

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
                      Output(DRAWDOWN_PARAM_4, 'options'),
                      Output(DRAWDOWN_PARAM_5, 'options'),
                      [Input(DRAWDOWN_RUN_ID, 'value')])
        def update_plot_parameter_names(data_id):
            paremeter_names = self._get_parameter_names_for_plot(data_id)
            paremeter_names = [{'label': i, 'value': i} for i in paremeter_names]
            return paremeter_names, paremeter_names, paremeter_names, paremeter_names, paremeter_names

    def _callback_select_data_options(self, app):
        @app.callback(Output(FIGURE_ID, 'figure'),
                      Input(DRAWDOWN_PARAM_1, 'value'),
                      Input(DRAWDOWN_PARAM_2, 'value'),
                      Input(DRAWDOWN_PARAM_3, 'value'),
                      Input(DRAWDOWN_PARAM_4, 'value'),
                      Input(DRAWDOWN_PARAM_5, 'value'),
                      Input(DRAWDOWN_PLOT_1, 'value'),
                      Input(DRAWDOWN_PLOT_2, 'value'),
                      Input(DRAWDOWN_PLOT_3, 'value'),
                      Input(DRAWDOWN_PLOT_4, 'value'),
                      Input(DRAWDOWN_PLOT_5, 'value'),
                      Input(DRAWDOWN_RUN_ID, 'value'),
                      Input(SECONDARY_CHECKLIST_ID, 'value'))
        def select_experiment(param_1,
                              param_2,
                              param_3,
                              param_4,
                              param_5,
                              plot_type_1,
                              plot_type_2,
                              plot_type_3,
                              plot_type_4,
                              plot_type_5,
                              data_id,
                              selected_secondary):
            figure = self._update_figure(param_1,
                                         param_2,
                                         param_3,
                                         param_4,
                                         param_5,
                                         plot_type_1,
                                         plot_type_2,
                                         plot_type_3,
                                         plot_type_4,
                                         plot_type_5,
                                         data_id,
                                         selected_secondary)
            if figure is None:
                raise PreventUpdate
            return figure

    def _get_parameter_names_for_plot(self, data_id: str) -> List[str]:
        return ['param11', 'param12', 'param13']

    def _get_data_id_list(self) -> List[str]:
        return ['run_id1', 'run_id2', 'run_id3']

    def _get_points(self, param_name, plot_type, secondary=False) -> go.Scatter:
        return None

    def _update_data_callback(self, data_id):
        pass

    def _update_data(self, data_id):
        if data_id != self.prev_data_id and not self.data_already_updated:
            self.data = None
            self.data_already_updated = True
            self.prev_data_id = data_id
            self._update_data_callback(data_id)
            self.data_already_updated = False
        while self.data_already_updated:
            time.sleep(0.5)

    def _update_figure(self,
                       param_1,
                       param_2,
                       param_3,
                       param_4,
                       param_5,
                       plot_type_1,
                       plot_type_2,
                       plot_type_3,
                       plot_type_4,
                       plot_type_5,
                       data_id,
                       selected_secondary) -> go.Figure:

        self._update_data(data_id)
        data = []
        for param, plot_type, sec_option_names in zip([param_1, param_2, param_3, param_4, param_5],
                                                      [plot_type_1, plot_type_2, plot_type_3, plot_type_4, plot_type_5],
                                                      CHECKLIST_OPTIONS):
            if param is not None and plot_type is not None:
                is_secondary_plot = selected_secondary is not None and sec_option_names in selected_secondary
                points = self._get_points(param, plot_type, is_secondary_plot)
                if points is not None:
                    data.append(points)
        layout = go.Layout(yaxis=dict(),
                           yaxis2=dict(overlaying='y',
                                       side='right'))
        return go.Figure(data=data, layout=layout)
