from readytofit.data.MlData import MlData
import json


class Model:

    """
    Model base class
    """

    def __init__(self, **kwargs):

        self.parameters = kwargs.get('parameters', {}).copy()
        self.random_batch = self.parameters.get('random_batch', True)

        self.meta = {}

    def fit(self, ml_data: MlData):

        """
        Fit model
        :param ml_data:
        :return:
        """
        self.meta['feature_names'] = list(ml_data.feature_names)

    def predict(self, ml_data: MlData):

        """
        Predict model
        :param ml_data:
        :return:
        """

    def predict_proba(self, ml_data):

        """
        Predict proba
        :param ml_data:
        :return:
        """

    def load(self, path):

        """
        Load model
        :param path:
        :return:
        """
        path_ = path + '.meta'
        with open(path_) as infile:
            self.meta = json.load(infile)

    def dump(self, path, model_name):

        """
        Dump model
        :param path:
        :param model_name:
        :return:
        """
        path_ = path + '/' + model_name + '.model.meta'
        if len(self.meta) > 0:
            with open(path_, "w") as outfile:
                json.dump(self.meta, outfile, indent=4)

    def dump_checkpoint(self, path, model_name, step):

        """

        :param path:
        :param model_name:
        :param step:
        :return:
        """

    def log_model(self):

        """
        Log model to logger
        :return:
        """

    def __str__(self):
        return self.__class__.__name__
