from readytofit.ModelCreator import ModelCreator, LabelCreator, FeatureCreator, CreatorApplyType, List, \
    DataSource, Optional, MetricBase, MlData, MLDatabaseManager, datetime, logged


@logged
class IncrementalModelCreator(ModelCreator):

    def __init__(self,
                 exp_id: str,
                 model_name: str,
                 label_creator: LabelCreator,
                 feature_creators: List[FeatureCreator],
                 data_source: DataSource,
                 metrics: Optional[List[MetricBase]] = None,
                 parameters: dict = None,
                 ml_database_manager: MLDatabaseManager = None,
                 run_id: datetime = None):
        ModelCreator.__init__(self,
                              exp_id,
                              model_name,
                              label_creator,
                              feature_creators,
                              data_source,
                              metrics,
                              parameters,
                              ml_database_manager,
                              run_id)

        once_feature_creators = list(filter(lambda x: x.once_all_data, feature_creators))
        if len(once_feature_creators) > 0:
            self._critical(f"incremental_learning doesn't support OnceOnAllData* "
                           f"features (there are {len(once_feature_creators)}), "
                           "because all features are creating before training")

        # def _train(self, ml_data: MlData, split_num: int = None):

