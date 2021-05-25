import dash
import dash_html_components as html
from tools import logged
from dashboard.MLDashboardManager import MlDashboardManager
from dashboard.exp_list_layout import exp_layout, exp_callback


@logged
class MlResultDesk:

    def __init__(self, *, ml_dashboard_manager: MlDashboardManager):
        external_stylesheets = ['/assets/dashboard.css', '/assets/trading_desk_dash.css']
        self._dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

        self._dash_app.layout = html.Div(exp_layout())
        exp_callback(self._dash_app, ml_dashboard_manager)

    def run(self):
        self._dash_app.run_server(debug=True, host='0.0.0.0', port="8060")
