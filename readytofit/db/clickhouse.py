from readytofit.tools.logging import logged
import pandas as pd
from readytofit.db.BaseDatabase import BaseDatabase
import clickhouse_driver


INSERT_QUERY = """INSERT INTO {}({}) VALUES"""
GET_QUERY = """SELECT * FROM {}"""
CREDS = "host='{}' dbname='{}' user='{}' password='{}' port='{}'"


@logged
class Clickhouse(BaseDatabase):

    def __init__(self, **kwargs):
        BaseDatabase.__init__(self, **kwargs)
        self.count_per_insert = kwargs.get('count_per_insert', 1000000)
        self.data_to_insert = {}
        self.client = clickhouse_driver.Client(user=self.user,
                                               password=self.password,
                                               database=self.database,
                                               port=self.port,
                                               host=self.host)
        self.client.connection.force_connect()

    def execute(self, query, data=None):

        try:
            if data is not None:
                self.client.execute(query, data, types_check=False)
            else:
                self.client.execute(query)

            self.client.disconnect()
        except BaseException as e:
            self._error(f'clickhouse exception: {e}')

    def insert_row(self, row, table, immediate_insert=False):

        data_class_types_count = len(set(map(type, row)))
        if data_class_types_count > 1:
            self._critical(f'You try pass more than one dataclass per insert. '
                           f'Unique data class types is {data_class_types_count}')
            return
        if len(row) == 0:
            return

        query = INSERT_QUERY.format(table,
                                    ','.join(row[0].__annotations__.keys()))
        row = list(map(lambda x: x.to_dict(), row))

        if immediate_insert:
            self.execute(query, row)
            return

        if query not in self.data_to_insert:
            self.data_to_insert[query] = []

        self.data_to_insert[query] += row
        if len(self.data_to_insert.get(query)) > self.count_per_insert:
            self.execute(query, self.data_to_insert.get(query))
            self.data_to_insert[query] = []

    def get_all(self, query=None, table=None):

        if query is None and table is None:
            self._critical('query and table are None')
        query = GET_QUERY.format(table) if query is None else query
        result = self.client.execute(query)
        self.client.disconnect()
        return result

    def __del__(self):

        for query in self.data_to_insert.keys():
            self.execute(query, self.data_to_insert.get(query))

        self.client.disconnect()

    @staticmethod
    def init_database(creds):
        if creds is not None:
            host, port, user, password, database = creds.split(':')
            db = Clickhouse(host=host,
                            port=port,
                            user=user,
                            password=password,
                            database=database)
            return db


def get_clickhouse_data(clickhouse_creds, symbol, values_count):

    host, port, user, password, database, table = clickhouse_creds.split(':')

    db = Clickhouse(host=host, port=port, user=user, password=password, database=database, table=table)

    query = """select price, timestamp 
        from market_data.ticks 
        where symbol == '{}' 
        order by tradeno desc limit {}"""
    result = db.get_all(query.format(symbol, values_count))
    result = pd.DataFrame(result, columns=['last', 'index'])
    return result
