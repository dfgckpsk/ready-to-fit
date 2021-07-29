from readytofit.data.MlData import MlData, List, np, pd
from readytofit.db.BaseDatabase import BaseDatabase


class DbMlData(MlData):

    def __init__(self,
                 feature_names: List[str],
                 database: BaseDatabase):
        self.feature_names = feature_names
        self.database = database
        self.table_name = ''
        self.index_name = 'index_name'

        self.cache_last_query = {}

        self.indexes = None

    def get_features(self, indexes=None, feature=None):
        if feature is None:
            feature = '*'
        elif type(feature) == list:
            feature = ','.join(feature)

        query = f"""select {feature} 
                    from {self.table_name} """
        if indexes is not None:
            query += f"""where {self.index_name} is in ({indexes})"""
        self.features = self.do_query(query)

    def do_query(self, query):
        if query not in self.cache_last_query.keys():
            self.cache_last_query[query] = self.database.get_all(query, self.table_name)
        return self.cache_last_query.get(query)

    def get_indexes(self):
        if self.indexes is None:
            query = f"""select {self.index_name} from {self.table_name}"""
            self.indexes = self.do_query(query)
        return self.indexes
