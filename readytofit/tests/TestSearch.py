from readytofit.search.GridSearch import GridSearch
from readytofit.search.RandomSearch import RandomSearch
from unittest import TestCase


class TestSearch(TestCase):

    def test_grid(self):

        config = {'param1': [1,2,3,4],
                  'param2': [11.1, 22.2, 33.3, 44.4],
                  'param3': ['str1', 'str2'],
                  'param4': ['value4'],
                  'param5': [100],
                  'param6': [10, 20]}

        gs = GridSearch(config)
        combinations = []
        for combination in gs.next_param():
            combinations.append(combination)

        self.assertEqual(len(combinations), 64)

    def test_random(self):

        config = {'param1': [1, 4],
                  'param2': [11.1, 44.4],
                  'param3': ['str1', 'str2'],
                  'param4': ['value4'],
                  'param5': [100],
                  'param6': [10, 20]}

        gs = RandomSearch(config, combinations_per_parameter=10)
        combinations = []
        for combination in gs.next_param():
            combinations.append(combination)

        assert all(map(lambda x: type(x['param1']) == int, combinations))
        assert all(map(lambda x: type(x['param2']) == float, combinations))
        assert all(map(lambda x: type(x['param3']) == str, combinations))
        assert all(map(lambda x: type(x['param4']) == str, combinations))
        assert all(map(lambda x: type(x['param5']) == int, combinations))
        assert all(map(lambda x: type(x['param6']) == int, combinations))
        self.assertEqual(len(combinations), 40)

        assert all(map(lambda x: 1 <= x['param1'] <= 4, combinations))
        assert all(map(lambda x: 11.1 <= x['param2'] <= 44.4, combinations))
        assert all(map(lambda x: x['param3'] in ('str1', 'str2'), combinations))
        assert all(map(lambda x: x['param4'] in ('value4'), combinations))
        assert all(map(lambda x: x['param5'] == 100, combinations))
        assert all(map(lambda x: 10 <= x['param6'] <= 20, combinations))