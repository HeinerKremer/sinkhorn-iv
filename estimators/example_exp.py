
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


class NetworkModel(torch.nn.Module):
    """A multilayer perceptron to approximate functions in the IV problem"""

    def __init__(self, kwargs=None):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(3, 1)
        )

    def forward(self, t):
        return self.model(t)


class NetworkIVData(Dataset):
    def __init__(self, n_sample, ftype='sin', dim_z=2):
        self.dim_z = dim_z
        self.func = self.set_function(ftype)
        self.data = self.generate_data(n_sample)
        super().__init__()

    def __getitem__(self, index):
        return {'t': self.data['t'][index], 'y': self.data['y'][index], 'z': self.data['z'][index]}

    def __len__(self):
        return len(self.data['t'])

    def generate_data(self, n_sample):
        """Generates train, validation and test data"""
        confounder = np.random.normal(loc=0, scale=1.0, size=[n_sample, 1])
        gamma = np.random.normal(loc=0, scale=0.1, size=[n_sample, 1])
        delta = np.random.normal(loc=0, scale=0.1, size=[n_sample, 1])

        z = np.random.uniform(low=-3, high=3, size=[n_sample, self.dim_z])
        t = np.reshape(z[:, 0], [-1, 1]) + confounder + gamma
        y = self.func(t) + confounder + delta
        return {"t": torch.FloatTensor(t), "y": torch.FloatTensor(y), "z": torch.FloatTensor(z)}

    def set_function(self, ftype):
        if ftype == 'linear':
            def func(x):
                return x
        elif ftype == 'sin':
            def func(x):
                return np.sin(x)
        elif ftype == 'step':
            def func(x):
                return np.asarray(x > 0, dtype=float)
        elif ftype == 'abs':
            def func(x):
                return np.abs(x)
        else:
            raise NotImplementedError
        return func

    def show_function(self, model=None, train_data=None,
                      test_data=None, title=''):
        t = test_data['t']

        g_true = self.func(t)
        g_test_pred = model.forward(t).detach().cpu().numpy()

        order = np.argsort(t[:, 0])
        t = t[:, 0]

        fig, ax = plt.subplots(1)
        ax.plot(t[order], g_true[order], label='True function', color='y')
        if train_data is not None:
            ax.scatter(train_data['t'][:, 0], train_data['y'], label='Data', s=6)

        if model is not None:
            ax.plot(t[order], g_test_pred[order], label='Model prediction', color='r')
        ax.legend()
        ax.set_title(title)
        plt.show()


    @staticmethod
    def get_model_factory():
        return NetworkModel

    def get_test_loss_funcs(self):
        def prediction_mse(y_pred, data):
            y_true = torch.FloatTensor(self.func(data['t'].detach().cpu().numpy()))
            return torch.nn.functional.mse_loss(y_pred.detach().cpu(), y_true)

        def fit_mse(y_pred, data):
            return torch.nn.functional.mse_loss(y_pred, data['y'])

        return {"validation_metric": fit_mse, "prediction_mse": prediction_mse, "fit_mse": fit_mse}


if __name__ == "__main__":
    dataset = NetworkIVData(1000, ftype="sin")
    data = next(iter(DataLoader(dataset, batch_size=1000)))

    model = dataset.get_model_factory()()
    dataset.show_function(model=model, train_data=data, test_data=data, title='True function and model prediction')
    