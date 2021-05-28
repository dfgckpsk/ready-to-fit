from readytofit.data.LabelCreator import LabelCreator, MlData


class AppendLabelCreator(LabelCreator):

    def __init__(self, append_value):
        self.append_value = append_value

    def get(self, data: MlData) -> MlData:
        labels = data.get_targets()
        labels += self.append_value
        data.update_labels(labels)
        return data
