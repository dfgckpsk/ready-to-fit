class BaseDatabase:

    def __init__(self, **kwargs):
        self.database = kwargs.get('database')
        self.host = kwargs.get('host')
        self.port = kwargs.get('port')
        self.user = kwargs.get('user')
        self.password = kwargs.get('password')

    def insert(self, data, table):
        if type(data) == list:
            for row in data:
                self.insert_row(row, table)

    def insert_row(self, row, table, immediate_insert=False):
        pass

    def get_all(self, query=None):
        pass
