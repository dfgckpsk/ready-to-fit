import dash


class DashboardObject:

    def __init__(self, app: dash.Dash):
        self.callback_objects = []
        self.app = app

    def layout(self):
        pass

    def callback(self):

        for callback_object in self.callback_objects:
            callback_object(self.app)
