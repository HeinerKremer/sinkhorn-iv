from collections import defaultdict

import numpy as np
import scipy.linalg
import torch

import utils


class KernelSMM:
    def __init__(self,
                 model,
                 datasets,
                 moment_function=None,
                 test_loss_funcs=None,
                 epsilon=1e-3,
                 reg_param=1e-6,
                 n_step=2):

        self.model = model
        self.datasets = datasets

        if not moment_function:
            def moment_function(y_pred, y_true):
                return y_pred - y_true

        self.moment_function = moment_function

        self.loss_funcs = test_loss_funcs
        self.n_step = n_step

        # Here `reg_param` corresponds to (\lambda * \gamma_t)/epsilon
        self.reg_param = reg_param
        self.epsilon = epsilon

        self.data = {}
        for mode, ds in datasets.items():
            self.data[mode] = utils.get_all_data(datasets)

        self.kernel_z = utils.get_rbf_kernel(self.data['train']['z'])
        self.weighting_matrix = None

        self.optimizer = torch.optim.LBFGS(self.model.parameters(), line_search_fn="strong_wolfe")
        self.loss_vals = {mode: defaultdict(list) for mode in self.datasets.keys()}

        # Infer the dimension of the moment function
        self.dim_psi = self.moment_function(self.model(self.data['train']['t'][:1], self.data['train']['y'][:1])).shape[1]

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

        for name, loss_func in self.loss_funcs.items():
            self.loss_vals[mode][name].append(loss_func(self.model, self.data['mode']))

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
        for i in range(self.dim_psi):
            hes = torch.autograd.functional.hessian(lambda t: self.moment_function(self.model(t), data['y']).sum(0)[i], inputs=t, create_graph=True)
            laplacian.append(torch.einsum("ijij -> i", hes))
        laplacian = torch.stack(laplacian, dim=1)
        return laplacian

    def _calc_weighting_matrix(self, data):
        n = self.kernel_z.shape[0]
        k_z_m = np.stack([self.kernel_z for _ in range(self.dim_psi)], axis=0)
        jac_psi = self.data_jac_per_sample(data).detach().cpu().numpy()

        # Indices chosen as in Theorem 3.7 of the paper
        q = np.einsum("lik, kls, krs, rjk -> lirj", k_z_m, jac_psi, jac_psi, k_z_m).reshape(self.dim_psi * n, self.dim_psi * n) / n
        del jac_psi

        l = scipy.linalg.block_diag(*k_z_m)
        del k_z_m
        q += self.reg_param * l
        try:
            weighting_matrix = l @ np.linalg.solve(q, l)
        except:
            weighting_matrix = l @ np.linalg.lstsq(q, l, rcond=None)[0]
        return torch.FloatTensor(weighting_matrix)


if __name__ == '__main__':
    pass