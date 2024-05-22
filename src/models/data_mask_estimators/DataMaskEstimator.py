import abc

from models.ClassificationModel import ClassProbabilities
from models.abstract_models.AbstractModel import Model


class DataMaskEstimator(Model):
    def __init__(self, dataset_name: str, x_dim: int, z_dim: int):
        super().__init__()
        if x_dim == 0 and z_dim == 0:
            raise Exception("cannot handle z_dim==0 and x_dim==0")
        self.use_z = z_dim > 0
        self.use_x = x_dim > 0
        self.z_dim = z_dim
        self.new_dataset_name = f"{dataset_name}_mask_use_z={self.use_z}_use_x={self.use_x}"
        self.correction = 0

    def fit(self, x_train, z_train, deleted_train, x_val, z_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            **kwargs):
        pass

    def predict(self, x, z):
        estimated_mask_probabilities = self.forward(x,z).probabilities[:, 1]
        estimated_mask_probabilities += self.correction
        return estimated_mask_probabilities

    def get_calibration_error(self, x_cal, z_cal, deleted_cal):
        estimated_mask_probabilities = self.forward(x_cal, z_cal).probabilities[:, 1]
        estimated_marginal_probability = estimated_mask_probabilities.mean().item()
        real_marginal_probability = deleted_cal.float().mean().item()
        return real_marginal_probability - estimated_marginal_probability

    def calibrate(self, x_cal, z_cal, deleted_cal, **kwargs):
        self.correction = self.get_calibration_error(x_cal, z_cal, deleted_cal)

    @abc.abstractmethod
    def forward(self, x, z) -> ClassProbabilities:
        pass

    def compute_performance(self, x_test, z_test, full_y_test, deleted_test) -> dict:
        probability_estimate = self.predict(x_test, z_test)
        mask_estimate = probability_estimate.round()
        accuracy = ((mask_estimate - deleted_test.float()).abs() < 1e-2).float().mean().item() * 100
        deleted_accuracy = ((mask_estimate - deleted_test.float()).abs() < 1e-2)[deleted_test].float().mean().item() * 100
        not_deleted_accuracy = ((mask_estimate - deleted_test.float()).abs() < 1e-2)[~deleted_test].float().mean().item() * 100
        ece = self.get_calibration_error(x_test, z_test, deleted_test)
        return {
            f"data_masker_accuracy": accuracy,
            f"data_masker_deleted_accuracy": deleted_accuracy,
            f"data_masker_~deleted_accuracy": not_deleted_accuracy,
            f"data_masker_correction": self.correction,
            f"data_masker_ECE": ece,
            f"data_masker_max_estimated_probability": probability_estimate.max().item(),
            f"data_masker_q99_estimated_probability": probability_estimate.quantile(q=0.99).item(),
            f"data_masker_q95_estimated_probability": probability_estimate.quantile(q=0.95).item(),
            f"data_masker_q9_estimated_probability": probability_estimate.quantile(q=0.9).item(),
            f"data_masker_q2_estimated_probability": probability_estimate.quantile(q=0.2).item(),
            f"data_masker_q1_estimated_probability": probability_estimate.quantile(q=0.1).item(),
        }