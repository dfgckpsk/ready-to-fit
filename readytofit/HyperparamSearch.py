from readytofit.ModelCreator import ModelCreator
from readytofit.data.LabelCreator import LabelCreator
from readytofit.data.FeatureCreator import FeatureCreator
from readytofit.metric.MetricBase import MetricBase
from readytofit.db.MLDatabaseManager import MLDatabaseManager
from readytofit.data.MlData import MlData
from readytofit.search.GridSearch import GridSearch
from readytofit.cv.CVBase import CVBase
from typing import List, Optional
from datetime import datetime


class HyperparamSearch:

    def __init__(self,
                 search_object: GridSearch,
                 exp_id: str,
                 model_name: str,
                 label_creator: LabelCreator,
                 feature_creators: List[FeatureCreator],
                 metrics: Optional[List[MetricBase]] = None,
                 ml_database_manager: MLDatabaseManager = None,
                 run_id: str = None,
                 cv_object: CVBase = None):
        self.exp_id = exp_id
        self.model_name = model_name
        self.search_object = search_object
        self.cv_object = cv_object
        self.metrics = metrics
        self.label_creator = label_creator
        self.feature_creators = feature_creators
        self.ml_database_manager = ml_database_manager
        self.run_id = run_id
        if self.run_id is None:
            self.run_id = datetime.utcnow()

    def run(self, ml_data: MlData):
        for parameters in self.search_object.next_param():
            model_creator = ModelCreator(exp_id=self.exp_id,
                                         model_name=self.model_name,
                                         label_creator=self.label_creator,
                                         feature_creators=self.feature_creators,
                                         metrics=self.metrics,
                                         parameters=parameters,
                                         ml_database_manager=self.ml_database_manager,
                                         run_id=self.run_id)
            if self.cv_object is None:
                model_creator.train(ml_data)
            else:
                model_creator.validation(ml_data, self.cv_object)
