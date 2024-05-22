import abc
import copy
import os.path

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import dill
from models.abstract_models.LearningModel import LearningModel
from plot_utils import display_plot
from utils import create_folder_if_it_doesnt_exist


class TransformDataset(TensorDataset):
    def __init__(self, *tensors: Tensor, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, item):
        data = super(TransformDataset, self).__getitem__(item)
        if type(data) == tuple:
            x = data[0]
        elif isinstance(data, torch.Tensor):
            x = data
        else:
            raise Exception(f"transformation {self} does not know how to handle with data={data}")
        if self.transform is not None:
            x = self.transform(x)

        return x, *data[1:]


def batch_transform(transform, x, batch_size=32):
    if len(x) > batch_size:
        pred = []
        for i in tqdm(range(0, len(x), batch_size), desc='batch transform'):
            start, end = i, min(i + batch_size, len(x))
            curr_pred = transform(x[start:end])
            pred += [curr_pred]
        pred = torch.cat(pred, dim=0)
    else:
        pred = transform(x)
    return pred


class NetworkLearningModel(LearningModel, nn.Module):
    def __init__(self, dataset_name: str, saved_models_path: str, figures_dir: str, seed: int):
        LearningModel.__init__(self, dataset_name, saved_models_path, seed)
        nn.Module.__init__(self)
        self._network = None
        self._optimizer = None
        self.figures_dir = figures_dir
        self.saved_models_path = saved_models_path
        self.dataset_name = dataset_name
        self.seed = seed
        self.fit_count = 0

    @property
    def network(self):
        return self._network

    @property
    def optimizer(self):
        return self._optimizer

    @abc.abstractmethod
    def loss(self, y, prediction, d, epoch):
        pass

    @abc.abstractmethod
    def predict(self, x, **kwargs):
        pass

    def fit_xy_aux(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64,
                   n_wait=20,
                   z_train=None,
                   z_val=None,
                   train_transform=None,
                   test_transform=None,
                   **kwargs):
        best_loss = np.inf
        epochs_without_improvement = 0
        train_losses = []
        val_losses = []
        network = self.network
        best_network = network
        if deleted_train is None:
            deleted_train = torch.zeros(len(x_train)).to(x_train.device)
        if deleted_val is None:
            deleted_val = torch.zeros(len(x_val)).to(x_val.device)
        if test_transform is not None:
            x_val = batch_transform(test_transform, x_val)
        use_z = z_train is not None
        if use_z:
            dataset = TransformDataset(x_train, z_train, y_train, deleted_train, transform=train_transform)
        else:
            dataset = TransformDataset(x_train, y_train, deleted_train, transform=train_transform)
        loader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=batch_size)
        for e in tqdm(range(epochs), desc='nn model fit'):
            epoch_losses = []
            self.train()
            for batch in loader:
                if use_z:
                    (x, z, y, d) = batch
                    z.requires_grad = True
                else:
                    z = None
                    (x, y, d) = batch
                x.requires_grad = True
                prediction = self.predict(x, z=z)
                curr_loss = self.loss(y, prediction, d, e)
                self.optimizer.zero_grad()
                curr_loss.backward()
                self.optimizer.step()
                epoch_losses += [curr_loss.item()]
            train_losses += [np.mean(epoch_losses)]

            with torch.no_grad():
                self.eval()
                prediction = self.predict(x_val, z=z_val)
                val_loss = self.loss(y_val, prediction, deleted_val, e).item()
                val_losses += [val_loss]

                if e > 110:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_network = copy.deepcopy(network)
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= n_wait:
                            break

        self._network = best_network
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        # import matplotlib
        # matplotlib.use('module://backend_interagg')

        save_dir = os.path.join(self.figures_dir, self.dataset_name, self.name, f'seed={self.seed}')
        display_plot(xs=[range(len(train_losses)), range(len(val_losses))],
                     ys=[train_losses, val_losses],
                     labels=['train', 'validation'],
                     title=f"Model {self.name} on data {self.dataset_name}",
                     x_label="Epoch", y_label="Loss", save_dir=save_dir)

        # corr(y_val[:, 1], y_val[:, 1] - prediction[1])
        # corr(y_val[deleted_val, 1], (y_val[:, 1] - prediction[1])[deleted_val])
        # corr(y_val[~deleted_val, 1], (y_val[:, 1] - prediction[1])[~deleted_val])

    def store_model(self):
        store_dir = self.get_model_save_dir()
        store_path = self.get_model_save_path()
        create_folder_if_it_doesnt_exist(store_dir)
        torch.save({
            'model_state_dict': self.state_dict(),
        },
            store_path, pickle_module=dill)

    def load_model(self):
        assert self.stored_model_exists()
        store_path = self.get_model_save_path()
        checkpoint = torch.load(store_path, map_location=lambda storage, loc: storage, pickle_module=dill)
        try:
            self.load_state_dict(checkpoint['model_state_dict'])
        except:
            self.network.load_state_dict(checkpoint['model_state_dict'])
