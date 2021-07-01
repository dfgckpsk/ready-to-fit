from readytofit.db.BaseDatabase import BaseDatabase
import redis
import pickle


class Redis(BaseDatabase):

    def __init__(self, **kwargs):
        BaseDatabase.__init__(self, **kwargs)
        self.redis = redis.StrictRedis(host=self.host,
                                       port=self.port,
                                       username=self.user,
                                       password=self.password,
                                       db=0)

    def insert(self, row, table):
        json_data = pickle.dumps(row)
        self.redis.set(table, json_data)

    def get_all(self, key, default=None):
        got_values = self.redis.get(key)
        if got_values is None:
            return default
        return pickle.loads(got_values)
