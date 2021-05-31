from readytofit.dashboard.UpdateTable import UpdateTable, dash, List
from readytofit.dashboard.MLDashboardData import MlDashboardData


class ExperimentsUpdateTable(UpdateTable):

    def __init__(self, app: dash.Dash, ml_dashboard_manager: MlDashboardData):
        UpdateTable.__init__(self, app)
        self.ml_dashboard_manager = ml_dashboard_manager
        self.title = 'Experiments'
        self.layout_columns = ['exp_id', 'uniq_count_runs', 'count_runs', 'min_date', 'max_date', 'max_split']

    def _get_table(self) -> List[dict]:
        runs = self.ml_dashboard_manager.get_exp_ids()
        runs = sorted(runs, reverse=True, key=lambda x: x.max_date)
        runs = list(map(lambda x: x.to_dict(), runs))
        return runs
