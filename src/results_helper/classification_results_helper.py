import torch

from models.ClassificationModel import ClassProbabilities, PredictionSets
from results_helper.results_helper import ResultsHelper


def compute_coverage(y: torch.Tensor, prediction_set: torch.Tensor):
    return prediction_set[range(len(y)), y.round().long()].float().mean().item()


def compute_length(prediction_set):
    return prediction_set.float().sum(dim=-1).mean().item()


class ClassificationResultsHelper(ResultsHelper):
    def __init__(self, base_results_save_dir, seed):
        super().__init__(base_results_save_dir, seed)

    def compute_performance_metrics_on_data_aux(self, full_y, y, deleted,
                                                model_prediction: ClassProbabilities,
                                                calibrated_uncertainty_sets: PredictionSets,) -> dict:
        full_y = full_y.squeeze()
        y = y.squeeze()
        assert len(full_y.shape) == len(y.shape) == 1
        estimated_probabilities = model_prediction.probabilities
        prediction_set = calibrated_uncertainty_sets.labels_one_hot.squeeze()
        y2_coverage = compute_coverage(y, prediction_set)
        not_deleted_y2_coverage = compute_coverage(full_y[~deleted], prediction_set[~deleted])
        deleted_y2_coverage = compute_coverage(full_y[deleted], prediction_set[deleted])
        full_y2_coverage = compute_coverage(full_y, prediction_set)
        y2_length = compute_length(prediction_set)
        deleted_y2_length = compute_length(prediction_set[deleted])
        not_deleted_y2_length = compute_length(prediction_set[~deleted])

        real_label_estimated_probability = estimated_probabilities[range(len(y)), full_y.round().long()]
        # probabilities = model_prediction.probabilities
        # y2_ece = compute_ece(y[:, 1], probabilities)
        # not_deleted_y2_ece = compute_ece(full_y[~deleted, 1], probabilities[~deleted])
        # deleted_y2_ece = compute_ece(full_y[deleted, 1], probabilities[deleted])
        # full_y2_ece = compute_ece(full_y[:, 1], probabilities)
        #
        # y2_oce = compute_oce(y[:, 1], probabilities)
        # not_deleted_y2_oce = compute_oce(full_y[~deleted, 1], probabilities[~deleted])
        # deleted_y2_oce = compute_oce(full_y[deleted, 1], probabilities[deleted])
        # full_y2_oce = compute_oce(full_y[:, 1], probabilities)
        #
        # model_certainty_level = compute_model_certainty(probabilities)
        # not_deleted_model_certainty_level = compute_model_certainty(probabilities[~deleted])
        # deleted_model_certainty_level = compute_model_certainty(probabilities[deleted])
        results = {
            'y2 coverage': y2_coverage,
            '~deleted y2 coverage': not_deleted_y2_coverage,
            'deleted y2 coverage': deleted_y2_coverage,
            'full y2 coverage': full_y2_coverage,
            'y2 length': y2_length,
            'deleted y2 length': deleted_y2_length,
            '~deleted y2 length': not_deleted_y2_length,
            'median max estimated probability': estimated_probabilities.max(dim=-1).values.median().item(),
            'deleted median max estimated probability': estimated_probabilities[deleted].max(dim=-1).values.median().item(),
            '~deleted median max estimated probability': estimated_probabilities[~deleted].max(dim=-1).values.median().item(),
            'q95 max estimated probability': estimated_probabilities.max(dim=-1).values.quantile(q=0.95).item(),
            'deleted q95 max estimated probability': estimated_probabilities[deleted].max(dim=-1).values.quantile(q=0.95).item(),
            '~deleted q95 max estimated probability': estimated_probabilities[~deleted].max(dim=-1).values.quantile(q=0.95).item(),
            'q9 q9 estimated probability': estimated_probabilities.quantile(q=0.9, dim=-1).quantile(q=0.9).item(),
            'deleted q9 q9 estimated probability': estimated_probabilities[deleted].quantile(q=0.9, dim=-1).quantile(q=0.9).item(),
            '~deleted q9 q9 estimated probability': estimated_probabilities[~deleted].quantile(q=0.9, dim=-1).quantile(q=0.9).item(),

            'q95 real label estimated probability': real_label_estimated_probability.quantile(q=0.95).item(),
            'deleted q95 real label estimated probability': real_label_estimated_probability[deleted].quantile(q=0.95).item(),
            '~deleted q95 real label estimated probability': real_label_estimated_probability[~deleted].quantile(q=0.95).item(),

            'q9 real label estimated probability': real_label_estimated_probability.quantile(q=0.9).item(),
            'deleted q9 real label estimated probability': real_label_estimated_probability[deleted].quantile(q=0.9).item(),
            '~deleted q9 real label estimated probability': real_label_estimated_probability[~deleted].quantile(q=0.9).item(),

            'q8 real label estimated probability': real_label_estimated_probability.quantile(q=0.8).item(),
            'deleted q8 real label estimated probability': real_label_estimated_probability[deleted].quantile(q=0.8).item(),
            '~deleted q8 real label estimated probability': real_label_estimated_probability[~deleted].quantile(q=0.8).item(),

            'q5 real label estimated probability': real_label_estimated_probability.quantile(q=0.5).item(),
            'deleted q5 real label estimated probability': real_label_estimated_probability[deleted].quantile(q=0.5).item(),
            '~deleted q5 real label estimated probability': real_label_estimated_probability[~deleted].quantile(q=0.5).item(),

            'q3 real label estimated probability': real_label_estimated_probability.quantile(q=0.3).item(),
            'deleted q3 real label estimated probability': real_label_estimated_probability[deleted].quantile(q=0.3).item(),
            '~deleted q3 real label estimated probability': real_label_estimated_probability[~deleted].quantile(q=0.3).item(),

            'q2 real label estimated probability': real_label_estimated_probability.quantile(q=0.2).item(),
            'deleted q2 real label estimated probability': real_label_estimated_probability[deleted].quantile(q=0.2).item(),
            '~deleted q2 real label estimated probability': real_label_estimated_probability[~deleted].quantile(q=0.2).item(),

            'q1 real label estimated probability': real_label_estimated_probability.quantile(q=0.2).item(),
            'deleted q1 real label estimated probability': real_label_estimated_probability[deleted].quantile(q=0.2).item(),
            '~deleted q1 real label estimated probability': real_label_estimated_probability[~deleted].quantile(q=0.2).item(),

            'q05 real label estimated probability': real_label_estimated_probability.quantile(q=0.05).item(),
            'deleted q05 real label estimated probability': real_label_estimated_probability[deleted].quantile(q=0.05).item(),
            '~deleted q05 real label estimated probability': real_label_estimated_probability[~deleted].quantile(q=0.05).item(),

            # 'y2 ece': y2_ece,
            # '~deleted y2 ece': not_deleted_y2_ece,
            # 'deleted y2 ece': deleted_y2_ece,
            # 'full y2 ece': full_y2_ece,
            #
            # 'y2 oce': y2_oce,
            # '~deleted y2 oce': not_deleted_y2_oce,
            # 'deleted y2 oce': deleted_y2_oce,
            # 'full y2 oce': full_y2_oce,
            #
            # 'model certainty level': model_certainty_level,
            # '~deleted model certainty level': not_deleted_model_certainty_level,
            # 'deleted model certainty level': deleted_model_certainty_level,
        }
        for label in torch.unique(full_y):
            label_coverage_rate = compute_coverage(full_y[full_y == label], prediction_set[full_y == label])
            label_deleted_coverage_rate = compute_coverage(full_y[(full_y == label) & deleted],
                                                           prediction_set[(full_y == label) & deleted])
            label_not_deleted_coverage_rate = compute_coverage(full_y[(full_y == label) & ~deleted],
                                                               prediction_set[(full_y == label) & ~deleted])
            results[f"label_{label}_coverage"] = label_coverage_rate
            results[f"label_deleted_{label}_coverage"] = label_deleted_coverage_rate
            results[f"label_~deleted_{label}_coverage"] = label_not_deleted_coverage_rate
        return results
