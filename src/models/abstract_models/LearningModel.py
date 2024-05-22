import abc
import os.path

from models.abstract_models.AbstractModel import Model
from utils import create_folder_if_it_doesnt_exist


class LearningModel(Model):
    def __init__(self, dataset_name: str, saved_models_path: str, seed: int):
        Model.__init__(self)
        self.saved_models_path = saved_models_path
        self.dataset_name = dataset_name
        self.seed = seed
        self.fit_count = 0

    @abc.abstractmethod
    def fit_xy_aux(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64,
                   n_wait=20,
                   **kwargs):
        pass

    def fit_xy(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
               **kwargs):
        if self.fit_count > 0:
            print(
                f"warning: {self.name} network learning model was fitted {self.fit_count} times already and is now fitted once again.")
        self.fit_count += 1

        if self.stored_model_exists():
            print(f"skipping fit of model {self.name} on data {self.dataset_name} because found stored model")
            self.load_model()
            self.eval()
            return
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(-1)
            y_val = y_val.unsqueeze(-1)

        print(f"staring fit of model {self.name} on data {self.dataset_name} for {epochs} epochs with bs={batch_size}")

        self.fit_xy_aux(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=epochs,
                        batch_size=batch_size, n_wait=n_wait, **kwargs)

        create_folder_if_it_doesnt_exist(self.get_model_save_dir())
        self.store_model()
        self.eval()

    def get_model_save_dir(self) -> str:
        return os.path.join(self.saved_models_path, self.dataset_name, self.save_name)

    def get_model_save_path(self) -> str:
        return os.path.join(self.get_model_save_dir(), f"seed={self.seed}.pth")

    def stored_model_exists(self):
        return os.path.exists(self.get_model_save_path())

    def eval(self):
        pass

    @abc.abstractmethod
    def store_model(self):
        pass

    @abc.abstractmethod
    def load_model(self):
        pass

    @property
    def save_name(self) -> str:
        return self.name
