import dash
import dash_html_components as html
from ml.tools import logged
from ml.dashboard.MLDashboardManager import MlDashboardManager
from ml.dashboard.exp_list_layout import exp_layout, exp_callback
from ml.dashboard.MLMetricsThreeVariablesPlot import MLMetricsThreeVariablesPlot


@logged
class MlResultDesk:

    def __init__(self, *, ml_dashboard_manager: MlDashboardManager):
        external_stylesheets = ['/assets/dashboard.css', '/assets/trading_desk_dash.css']
        self._dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        tvp = MLMetricsThreeVariablesPlot(ml_dashboard_manager)

        self._dash_app.layout = html.Div(exp_layout() + tvp.layout())
        exp_callback(self._dash_app, ml_dashboard_manager)
        tvp.callback(self._dash_app)

    def run(self):
        self._dash_app.run_server(debug=True, host='0.0.0.0', port="8060")
