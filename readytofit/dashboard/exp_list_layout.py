import dash_core_components as dcc
import dash_html_components as html
import dash_table as dtable
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from readytofit.dashboard.MLDashboardData import MlDashboardData
from readytofit.db.CreatedMlModels import CreatedMlModels

EXP_ID_TABLE = 'experiments-id'
UPDATE_RUNS_ID = 'update-runs-button'
EXP_DRAWDOWN_ID = 'exp-drawdown-id'
RUN_ID_TABLE = 'run-table-id'
RUN_CHECKLIST_ID = 'run-checklist-id'
# dcc.Dropdown


def exp_layout():
    experiements_layout_columns = ['exp_id', 'uniq_count_runs', 'count_runs', 'min_date', 'max_date', 'max_split']
    experiements_layout_columns = list(map(lambda x: {'name': x, 'id': x}, experiements_layout_columns))

    run_columns = ['exp_id', 'run_id']
    run_columns = list(map(lambda x: {'name': x, 'id': x}, run_columns))

    run_checklist_values = CreatedMlModels.get_field_names()
    run_checklist_values = list(map(lambda x: {'label': x, 'value': x}, run_checklist_values))

    return [html.H2('Experiments'),
            dcc.Loading(
                id="loading-1",
                type="circle",
                children=html.Div(id="loading-output-1")
            ),
            html.Button('Update runs', id=UPDATE_RUNS_ID, n_clicks=0),
            dtable.DataTable(id=EXP_ID_TABLE,
                             row_selectable=False,
                             editable=False,
                             columns=experiements_layout_columns,
                             page_size=12,
                             style_cell={'whiteSpace': 'normal',
                                         'height': 'auto'}),
            html.H2('Runs'),
            dcc.Dropdown(id=EXP_DRAWDOWN_ID,
                         style={'width': '45%', 'display': 'inline-block'}),
            html.H3('Checklist columns'),
            dcc.Checklist(id=RUN_CHECKLIST_ID,
                          options=run_checklist_values,
                          value=['run_id', 'exp_id'],
                          labelStyle={'display': 'inline-block'}),
            dcc.Loading(
                id="loading-2",
                type="circle",
                children=html.Div(id="loading-output-2")
            ),
            dtable.DataTable(id=RUN_ID_TABLE,
                             row_selectable=False,
                             editable=False,
                             columns=run_columns,
                             filter_action="native",
                             sort_action="native",
                             sort_mode="multi",
                             page_size=12,
                             style_cell={'whiteSpace': 'normal',
                                         'height': 'auto'})
            ]


def exp_callback(app, ml_dashboard_manager: MlDashboardData):

    @app.callback(Output(EXP_ID_TABLE, 'data'),
                  Output("loading-output-1", "children"),
                  Output(EXP_DRAWDOWN_ID, 'options'),
                  [Input(UPDATE_RUNS_ID, 'n_clicks')])
    def update_experiments(n_clicks):
        runs = ml_dashboard_manager.get_exp_ids()
        # print(runs)
        runs = sorted(runs, reverse=True,  key=lambda x: x.max_date)
        exp_ids = list(map(lambda x: x.exp_id, runs))
        runs = list(map(lambda x: x.to_dict(), runs))
        return runs, True, [{'label': i, 'value': i} for i in exp_ids]

    @app.callback(Output(RUN_ID_TABLE, 'data'),
                  Output(RUN_ID_TABLE, 'columns'),
                  Output("loading-output-2", "children"),
                  Input(EXP_DRAWDOWN_ID, 'value'),
                  Input(RUN_CHECKLIST_ID, 'value'))
    def select_experiment(exp_id, columns):
        global experiments
        if exp_id is None:
            raise PreventUpdate()
        experiments = ml_dashboard_manager.get_experiment_results(exp_id)
        experiments = list(map(lambda x: x.to_dict_str(), experiments))
        columns = list(map(lambda x: {'name': x, 'id': x}, columns))

        return experiments, columns, True
