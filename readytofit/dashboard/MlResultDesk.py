import dash
import dash_html_components as html
from readytofit.tools import logged
from readytofit.dashboard.MLDashboardData import MlDashboardData
from readytofit.dashboard.exp_list_layout import exp_layout, exp_callback
from readytofit.dashboard.MLMetricsThreeVariablesPlot import MLMetricsThreeVariablesPlot
from readytofit.dashboard.ExperiementsUpdateTable import ExperimentsUpdateTable


@logged
class MlResultDesk:

    def __init__(self, *, ml_dashboard_data: MlDashboardData):
        external_stylesheets = ['/assets/dashboard.css', '/assets/trading_desk_dash.css']
        self._dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        eut = ExperimentsUpdateTable(self._dash_app, ml_dashboard_data)
        tvp = MLMetricsThreeVariablesPlot(self._dash_app, ml_dashboard_data)

        self._dash_app.layout = html.Div(eut.layout() + tvp.layout())
        # exp_callback(self._dash_app, ml_dashboard_data)
        eut.callback()
        tvp.callback()

    def run(self):
        self._dash_app.run_server(debug=True, host='0.0.0.0', port="8060")
