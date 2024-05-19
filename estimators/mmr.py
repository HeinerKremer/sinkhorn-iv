from collections import defaultdict

import numpy as np
import scipy.linalg
import torch
from scipy.spatial.distance import cdist


class MMR:
    def __init__(self, 
                 model, 
                 datasets,
                 test_loss_funcs=None):
            
        self.model = model
        self.datasets = datasets
        self.loss_funcs = test_loss_funcs if test_loss_funcs else {}
        self.moment_function = lambda y_pred, y_true: y_pred - y_true

        self.data = {}
        for mode, ds in datasets.items():
        # Collect all data at once as MMR is a full batch method
            self.data[mode] = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)))
        self.kernel_z, _ = get_rbf_kernel(self.data['train']['z'])

        self.optimizer = torch.optim.LBFGS(self.model.parameters(), line_search_fn="strong_wolfe")
        self.loss_vals = {mode: defaultdict(list) for mode in self.datasets.keys()}

    def objective(self, data):
        psi = self.moment_function(self.model(data['t']), data['y'])
        loss = torch.einsum('ir, ij, jr -> ', psi, self.kernel_z, psi) / (len(psi) ** 2)
        return loss
    
    def fit(self):
        for mode in self.data.keys():
            self._epoch(mode=mode)
        return self.loss_vals

    def _epoch(self, mode):
        if mode == 'train':

            def closure():
                self.optimizer.zero_grad()
                obj = self.objective(self.data['train'])
                obj.backward()
                return obj

            self.optimizer.step(closure)
            self.loss_vals[mode]['objective'].append(self.objective(self.data[mode]))

        y_pred = self.model(self.data[mode]['t'])
        for name, loss_func in self.loss_funcs.items():
            self.loss_vals[mode][name].append(loss_func(y_pred, self.data[mode]))


def get_rbf_kernel(x_1, x_2=None, sigma=None, numpy=False):
    if x_2 is None:
        x_2 = x_1

    n_1, n_2 = x_1.shape[0], x_2.shape[0]
    sq_dist = torch.cdist(torch.reshape(x_1, (n_1, -1)), torch.reshape(x_2, (n_2, -1))) ** 2

    if sigma is None and numpy:
        median = np.median(sq_dist.flatten()) ** 0.5
        sigma = median
    elif sigma is None and not numpy:
        sigma = torch.median(sq_dist) ** 0.5

    kernel_zz = torch.exp((-1 / (2 * sigma ** 2)) * sq_dist)
    if numpy:
        kernel_zz = kernel_zz.detach().numpy()
    return kernel_zz, sigma


if __name__ == '__main__':
    from example_exp import NetworkIVData

    dataset = NetworkIVData(1000, ftype="step")
    all_data = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    model = dataset.get_model_factory()()
    smm = MMR(model, {"train": dataset}, test_loss_funcs=dataset.get_test_loss_funcs())
    smm.fit()
    dataset.show_function(model, test_data=all_data, train_data=all_data, title="Kernel-SMM with sin function")