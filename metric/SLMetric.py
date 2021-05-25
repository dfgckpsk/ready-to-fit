from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, mean_absolute_error
from metric.MetricBase import MetricBase
from tools.logging import logged
import numpy as np


@logged
class SLMetric(MetricBase):

    def __init__(self, metric_name):
        self.metric_name = metric_name

    def calculate(self, real_values, predicted_values):

        if self.metric_name == 'accuracy':
            return accuracy_score(real_values, predicted_values.reshape(-1))
        elif self.metric_name == 'f1_avg_none':
            return f1_score(real_values, predicted_values.reshape(-1), average=None)
        elif self.metric_name == 'f1_macro':
            return f1_score(real_values, predicted_values.reshape(-1), average='macro')
        elif self.metric_name == 'mae':
            return mean_absolute_error(real_values, predicted_values.reshape(-1))
        elif self.metric_name == 'label_mean':
            return np.mean(predicted_values.reshape(-1))
        elif self.metric_name == 'mse':
            return mean_squared_error(real_values, predicted_values.reshape(-1))
        elif self.metric_name == 'f1_micro':
            return f1_score(real_values, predicted_values.reshape(-1), average='micro')
        elif self.metric_name == 'f1_weighted':
            return f1_score(real_values, predicted_values.reshape(-1), average='weighted')
        elif self.metric_name == 'f1_samples':
            return f1_score(real_values, predicted_values.reshape(-1), average='samples')
        elif self.metric_name == 'classification_report':
            c = classification_report(real_values, predicted_values.reshape(-1), output_dict=True)
            return classification_report(real_values, predicted_values.reshape(-1), output_dict=True)
        elif self.metric_name == 'precision_macro_avg':
            c = classification_report(real_values, predicted_values.reshape(-1), output_dict=True)
            return c['macro avg']['precision']
        elif self.metric_name == 'recall_macro_avg':
            c = classification_report(real_values, predicted_values.reshape(-1), output_dict=True)
            return c['macro avg']['recall']
        elif self.metric_name == 'precision_weighted_avg':
            c = classification_report(real_values, predicted_values.reshape(-1), output_dict=True)
            return c['weighted avg']['precision']
        elif self.metric_name == 'recall_weighted_avg':
            c = classification_report(real_values, predicted_values.reshape(-1), output_dict=True)
            return c['weighted avg']['recall']

        elif self.metric_name == 'recall_avg_none':
            c = classification_report(real_values, predicted_values.reshape(-1), output_dict=True)
            classes = sorted(list(filter(lambda x: len(x) == 1, list(c.keys()))))
            recalls = list(map(lambda x: c[x]['recall'], classes))
            return recalls

        elif self.metric_name == 'precision_avg_none':
            c = classification_report(real_values, predicted_values.reshape(-1), output_dict=True)
            classes = sorted(list(filter(lambda x: len(x) == 1, list(c.keys()))))
            precisions = list(map(lambda x: c[x]['precision'], classes))
            return precisions

        self._warning(f'Metric {self.metric_name} is not implemented')

    def __str__(self):
        return self.metric_name
