import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class DataScaler():

    def __init__(self):

        self.s_tr_x = StandardScaler()
        self.s_tr_y = StandardScaler()
        self.s_tr_z = StandardScaler()

    def initialize_scalers(self, x_train, y_train=None, z_train=None):
        if len(x_train.shape) == 2:
            self.s_tr_x = self.s_tr_x.fit(x_train.cpu())
        if y_train is not None:
            y_train = y_train.unsqueeze(-1) if len(y_train.shape) == 1 else y_train
            self.s_tr_y = self.s_tr_y.fit(y_train.cpu())
        if z_train is not None:
            self.s_tr_z = self.s_tr_z.fit(z_train.cpu())

    @staticmethod
    def scale(scaler: StandardScaler, *tensor_list):
        scaled_t = []
        for t in tensor_list:
            t = t.unsqueeze(-1) if len(t.shape) == 1 else t
            if len(t.shape) == 2:
                scaled_t += [torch.Tensor(scaler.transform(t.detach().cpu())).to(t.device)]
            else:
                scaled_t += [t]

        if len(scaled_t) == 1:
            scaled_t = scaled_t[0]
        return scaled_t

    @staticmethod
    def unscale(scaler: StandardScaler, t):
        if len(t.shape) > 2:
            return t
        if len(t.shape) == 1:
            squeeze = True
            t = t.unsqueeze(-1)
        else:
            squeeze = False

        res = torch.Tensor(scaler.inverse_transform(t.detach().cpu())).to(t.device)
        if squeeze:
            res = res.squeeze()

        return res

    def scale_x(self, *x_list):
        return DataScaler.scale(self.s_tr_x, *x_list)

    def scale_y(self, *y_list):
        return DataScaler.scale(self.s_tr_y, *y_list)

    def scale_z(self, *z_list):
        return DataScaler.scale(self.s_tr_z, *z_list)

    def unscale_y(self, y):
        y = y.unsqueeze(-1) if len(y.shape) == 1 else y
        res = torch.Tensor(self.s_tr_y.inverse_transform(y.detach().cpu())).to(y.device).squeeze()
        return res

    def unscale_z(self, z):
        return DataScaler.unscale(self.s_tr_z, z)

    def scale_x_y(self, x, y):
        x, y = self.scale_x(x), self.scale_y(y)
        return x, y

    def unscale_x(self, x):
        return DataScaler.unscale(self.s_tr_x, x)

