from typing import List
from readytofit.dashboard.base.DashboardObject import DashboardObject, dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dtable
from dash.dependencies import Input, Output

EXP_ID_TABLE = 'experiments-id'
UPDATE_RUNS_ID = 'update-runs-button'


class UpdateTable(DashboardObject):

    def __init__(self, app: dash.Dash):
        DashboardObject.__init__(self, app)
        self.callback_objects.append(self._callback)
        self.experiements_layout_columns = []
        self.title = 'title'

    def layout(self):
        experiements_layout_columns = list(map(lambda x: {'name': x, 'id': x}, self.experiements_layout_columns))
        return [html.H2(self.title),
        html.Button('Update', id=UPDATE_RUNS_ID, n_clicks=0),
        dcc.Loading(
            id="loading-1",
            type="circle",
            children=dtable.DataTable(id=EXP_ID_TABLE,
                         row_selectable=False,
                         editable=False,
                         columns=experiements_layout_columns,
                         page_size=12,
                         style_cell={'whiteSpace': 'normal',
                                     'height': 'auto'})
        )]

    def _callback(self, app):
        @app.callback(Output(EXP_ID_TABLE, 'data'),
                      [Input(UPDATE_RUNS_ID, 'n_clicks')])
        def update_experiments(n_clicks):
            data = self._get_table()
            return data

    def _get_table(self) -> List[dict]:
        return []
