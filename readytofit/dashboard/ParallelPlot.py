from ..dashboard.base.DashboardObject import DashboardObject
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash
from typing import List

FIGURE_ID = 'parallel-figure-id'
LOADING_ID_1 = 'parallel-loading-1'
LOADING_ID_2 = 'parallel-loading-2'
DATA_ID_DRAWDOWN = 'parallel-data-id-drawdown'
MAIN_PARAMETERS_DRAWDOWN = 'main-parameters_drawdown'
LINE_PARAMETERS_DRAWDOWN = 'line-parameters_drawdown'
UPDATE_RUNS_ID = 'parallel-update-runs-id'


class ParallelPlot(DashboardObject):

    def __init__(self, app: dash.Dash, title: str = 'ParallelPlot'):
        DashboardObject.__init__(self, app)
        self.callback_objects = [self._callback_update_data_id,
                                 self._callback_selected_parameters]
        self.title = title
        self.data = {}

    def layout(self):

        layout = []

        layout += [html.H2(self.title),
                   html.Button('Update runs', id=UPDATE_RUNS_ID, n_clicks=0)]
        figure_layout = self._layout_figure()
        parameters_layouts = self._layout_parameter_select()
        data_layout = html.Div([html.Div(figure_layout, style={'width': '80%', 'display': 'inline-block'}),
                                html.Div(parameters_layouts, style={'width': '20%',
                                                                    'display': 'inline-block', 'vertical-align': 'top'})],
                               style={'width': '100%', 'display': 'inline-block'})
        return [html.Div(layout), data_layout]

    def _layout_figure(self):
        return [dcc.Loading(id=LOADING_ID_1,
                            type="circle",
                            children=dcc.Graph(id=FIGURE_ID, style={'height': '90vh'}))]

    def _layout_parameter_select(self):
        drawdowns = [dcc.Dropdown(id=MAIN_PARAMETERS_DRAWDOWN, multi=True),
                     dcc.Dropdown(id=LINE_PARAMETERS_DRAWDOWN)]
        return [html.H4('Parameters'),
                dcc.Dropdown(id=DATA_ID_DRAWDOWN),
                dcc.Loading(id=LOADING_ID_2,
                            type="circle",
                            children=drawdowns)]

    def _get_main_params(self) -> List[str]:
        return list(self.data.keys())

    def _get_line_params(self) -> List[str]:
        return list(self.data.keys())

    def _callback_update_data_id(self, app):
        @app.callback(Output(MAIN_PARAMETERS_DRAWDOWN, 'options'),
                      Output(LINE_PARAMETERS_DRAWDOWN, 'options'),
                      Input(DATA_ID_DRAWDOWN, 'value'))
        def update_parameters(data_id):
            self._update_data(data_id)
            main_parameters = self._get_main_params()
            main_parameters = [{'label': i, 'value': i} for i in main_parameters]
            line_parameters = self._get_line_params()
            line_parameters = [{'label': i, 'value': i} for i in line_parameters]
            return main_parameters, line_parameters

        @app.callback(Output(DATA_ID_DRAWDOWN, 'options'),
                      Input(UPDATE_RUNS_ID, 'value'))
        def update_run_id(n_clicks):
            run_ids = self._get_data_id_list()
            run_ids = [{'label': i, 'value': i} for i in run_ids]
            return run_ids

    def _callback_selected_parameters(self, app):
        @app.callback(Output(FIGURE_ID, 'figure'),
                      Input(MAIN_PARAMETERS_DRAWDOWN, 'value'),
                      Input(LINE_PARAMETERS_DRAWDOWN, 'value'))
        def update_parameters(main_parameters,
                              line_parameter):
            return self._update_figure(main_parameters, line_parameter)

    def _get_data_id_list(self) -> List[str]:
        return ['data_id_1', 'data_id_2']

    def _update_figure(self,
                       main_parameters,
                       line_parameter) -> go.Figure:
        kwargs = {}

        if line_parameter is not None:
            data = self._get_data(line_parameter)
            if data is not None:
                kwargs['line'] = dict(color=data,
                                      showscale=True)

        if main_parameters is None:
            raise PreventUpdate

        dimensions = []
        for param in main_parameters:
            data = self._get_data(param)
            if data is not None:
                dimensions.append(dict(label=param,
                                       values=data))
        kwargs['dimensions'] = dimensions

        return go.Figure(data=go.Parcoords(**kwargs))

    def _update_data(self, data_id):
        import pandas as pd
        df = pd.read_csv("https://raw.githubusercontent.com/bcdunbar/datasets/master/parcoords_data.csv")
        self.data = {}
        for col in df.columns:
            self.data[col] = list(df[col].values)

    def _get_data(self, name):
        return self.data.get(name)
