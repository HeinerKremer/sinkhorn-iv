from collections import defaultdict

import numpy as np
import scipy.linalg
import torch


class KernelSMM:
    def __init__(self,
                 model,
                 datasets,
                 test_loss_funcs=None,
                 epsilon=1e-2,
                 reg_param=1e-6,
                 n_step=2):
        """
        Initialize the SMMKernel estimator.

        Args:
            model: The model to be trained.
            datasets: A dictionary containing the training, validation, and test datasets with keys {'train', 'val', 'test'}.
            test_loss_funcs: A dictionary containing the loss functions for testing.
            epsilon: The regularization parameter epsilon in the Sinkorn distance.
            reg_param: The regularization parameter (\lambda * \gamma_t)/epsilon.
            n_step: The number of steps for estimation.


        """

        self.model = model
        self.datasets = datasets
        self.loss_funcs = test_loss_funcs if test_loss_funcs else {}
        self.loss_vals = {mode: defaultdict(list) for mode in self.datasets.keys()}
        self.n_step = n_step

        # Here `reg_param` corresponds to (\lambda * \gamma_t)/epsilon
        self.reg_param = reg_param
        self.epsilon = epsilon

        self.data = {}
        for mode, ds in datasets.items():
            # Collect all data at once as kernel-SMM is a full batch method
            self.data[mode] = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)))

        self.kernel_z, _ = get_rbf_kernel(self.data['train']['z'])
        self.weighting_matrix = None

        self.optimizer = torch.optim.LBFGS(self.model.parameters(), line_search_fn="strong_wolfe")

        self.moment_function = lambda y_pred, y_true: y_pred - y_true
        self.dim_moment_function = self.data['train']['y'].shape[1]

    def fit(self):
        for step in range(self.n_step):
            for mode in self.data.keys():
                self._epoch(mode=mode)
        return self.loss_vals

    def _epoch(self, mode):
        if mode == 'train':
            self.weighting_matrix = self._calc_weighting_matrix(self.data['train'])

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

    def objective(self, data):
        moment_func = self.moment_function(self.model(data['t']), data['y'])
        psi_x = (moment_func + self.epsilon / 2 * self.data_laplacian_per_sample(data)).transpose(1, 0).flatten()
        weighted_psi = torch.matmul(self.weighting_matrix, psi_x).detach()
        loss = torch.matmul(weighted_psi, psi_x) / (len(moment_func)**2)
        return loss

    def data_jac_per_sample(self, data):
        """
        Derivatives only with respect to t because Kernel-SMM fixes the instruments z and for IV psi = y - f(x),
        so the gradient with respect to y is the same as the regularization term \| h \|^2.
        """
        t = torch.nn.Parameter(data['t'], requires_grad=True)
        jac = torch.autograd.functional.jacobian(lambda t: self.moment_function(self.model(t), data['y']), inputs=t, create_graph=True)
        jac = torch.einsum("nynt -> nyt", jac)
        return jac

    def data_laplacian_per_sample(self, data):
        """
        Derivatives only with respect to t because Kernel-SMM fixes the instruments z and for IV psi = y - f(x),
        so the laplacian with respect to y will always be 0.
        """
        t = torch.nn.Parameter(data['t'], requires_grad=True)
        laplacian = []
        for i in range(self.dim_moment_function):
            hes = torch.autograd.functional.hessian(lambda t: self.moment_function(self.model(t), data['y']).sum(0)[i], inputs=t, create_graph=True)
            laplacian.append(torch.einsum("ijij -> i", hes))
        laplacian = torch.stack(laplacian, dim=1)
        return laplacian

    def _calc_weighting_matrix(self, data):
        n = self.kernel_z.shape[0]
        k_z_m = np.stack([self.kernel_z for _ in range(self.dim_moment_function)], axis=0)
        jac_psi = self.data_jac_per_sample(data).detach().cpu().numpy()

        # Indices chosen as in Theorem 3.7 of the paper
        q = np.einsum("lik, kls, krs, rjk -> lirj", k_z_m, jac_psi, jac_psi, k_z_m).reshape(self.dim_moment_function * n, self.dim_moment_function * n) / n
        del jac_psi

        l = scipy.linalg.block_diag(*k_z_m)
        del k_z_m
        q += self.reg_param * l
        try:
            weighting_matrix = l @ np.linalg.solve(q, l)
        except:
            weighting_matrix = l @ np.linalg.lstsq(q, l, rcond=None)[0]
        return torch.FloatTensor(weighting_matrix)


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
    smm = KernelSMM(model, {"train": dataset}, test_loss_funcs=dataset.get_test_loss_funcs())
    smm.fit()
    dataset.show_function(model, test_data=all_data, train_data=all_data, title="Kernel-SMM with sin function")