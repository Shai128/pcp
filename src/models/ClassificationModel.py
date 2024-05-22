import abc
import torch

from models.abstract_models.AbstractModel import Model
from models.model_utils import UncertaintySets, ModelPrediction


class ClassProbabilities(ModelPrediction):

    def __init__(self, probabilities: torch.Tensor):
        super().__init__()
        self.probabilities = probabilities

    def __getitem__(self, item):
        return ClassProbabilities(self.probabilities.__getitem__(item))


class PredictionSets(UncertaintySets):

    def __init__(self, labels_one_hot: torch.Tensor):
        self.labels_one_hot = labels_one_hot

    def __getitem__(self, item):
        return PredictionSets(self.labels_one_hot.__getitem__(item))

    def contains(self, y):
        return self.labels_one_hot[y.long()]

    def union(self, other_set: UncertaintySets) -> UncertaintySets:
        if not isinstance(other_set, PredictionSets):
            raise Exception(f"cannot union PredictionSets with {type(other_set)}")
        new_labels_one_hot = self.labels_one_hot.clone()
        new_labels_one_hot = new_labels_one_hot | other_set.labels_one_hot
        return PredictionSets(new_labels_one_hot)


class ClassificationModel(Model):

    @abc.abstractmethod
    def estimate_probabilities(self, x: torch.Tensor) -> ClassProbabilities:
        pass

    def fit(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            **kwargs):
        pass

    @abc.abstractmethod
    def eval(self):
        pass

    def compute_performance(self, x_test, z_test, full_y_test, deleted_test):
        probabilities = self.estimate_probabilities(x_test).probabilities
        return {
            'classification_loss': torch.nn.functional.cross_entropy(probabilities, full_y_test.long().squeeze()).item()
        }