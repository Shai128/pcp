# import torch
#
# from data_utils.datasets.synthetic_dataset_generator import SyntheticDataGenerator
# from utils import set_seeds, get_seed
#
# # from data_utils.get_dataset_utils import max_val1, max_val2, threshold
# from data_utils.data_masker import DataMaskingInfo
#
#
# class ClassificationSyntheticDataGenerator(SyntheticDataGenerator):
#     def __init__(self, data_size: int, x_dim: int, n_classes: int):
#         super().__init__(data_size)
#         self.data_size = data_size
#         self.x_dim = x_dim
#         curr_seed = get_seed()
#         set_seeds(0)
#         self.beta_vectors = torch.randn(x_dim, n_classes)
#         self.n_classes = n_classes
#         set_seeds(curr_seed)
#
#     def generate_data(self, device='cpu'):
#         curr_seed = get_seed()
#         set_seeds(0)
#
#         x = torch.randn(self.data_size, self.x_dim)
#         z = torch.exp(x @ self.beta_vectors)
#         y_prob = (z.T / z.sum(dim=-1)).T
#         y = torch.multinomial(y_prob, 1).squeeze().float()
#
#         # [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
#         # np.argsort([abs(np.corrcoef(X[:, i], y)[0,1]) for i in range(X.shape[1])])
#         # import matplotlib
#         # matplotlib.use('module://backend_interagg')
#         # plt.scatter(x[:, 3], y)
#         # plt.xlabel("x")
#         # plt.ylabel("y")
#         # plt.show()
#
#         most_correlated_x_idx = 3
#         most_correlated_x = x[:, most_correlated_x_idx]
#         indicator = (torch.rand_like(y) <= 0.4).float()
#         new_y = (torch.max(torch.min(most_correlated_x.abs().round() * 4, torch.Tensor([self.n_classes - 1])),
#                            torch.Tensor([0])))
#         y = y * (1 - indicator) + new_y.float() * indicator
#
#         max_val1 = torch.quantile(most_correlated_x, q=0.9).item()
#         max_val2 = torch.quantile(most_correlated_x, q=0.85).item()
#         threshold = torch.quantile(most_correlated_x, q=0.7).item()
#
#         DataMaskingInfo.max_val1 = max_val1
#         DataMaskingInfo.max_val2 = max_val2
#         DataMaskingInfo.threshold = threshold
#
#         vals = most_correlated_x.clone()
#         vals[vals >= max_val2] = max_val2
#         vals /= max_val1
#         vals[most_correlated_x < threshold] = 0
#         probability_to_delete = vals  # ** 0.8
#         deleted = torch.rand_like(most_correlated_x) < probability_to_delete
#         deleted.float().mean()
#         proxy = most_correlated_x.squeeze().clone()
#
#         x[:, most_correlated_x_idx] = x[:, most_correlated_x_idx] + 0.3 * torch.randn_like(x[:, most_correlated_x_idx])
#         set_seeds(curr_seed)
#
#         x, y, deleted = x.to(device), y.to(device), deleted.to(device)
#         proxy = proxy.to(device)
#         y = torch.cat([proxy.unsqueeze(-1), y.unsqueeze(-1)], dim=-1).float()
#         return x, y, deleted
