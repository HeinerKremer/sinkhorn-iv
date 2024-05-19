import numpy as np
import torch
from collections import defaultdict
import multiprocessing
from tqdm import tqdm


class NeuralSMM:
    def __init__(self,
                     model,
                     datasets,
                     val_loss_func=None,
                     test_loss_funcs=None,
                     epsilon=1e-2,
                     reg_param=1e-1,
                     dual_model=None,
                     lr_model=5e-4,
                     lr_dual=5e-4,
                     batch_size=200,
                     epochs=3000,
                     patience=10,
                     device='cuda'):
            """
            Parameters:
            - model: The model to be trained.
            - datasets: A dictionary containing the training, validation, and test datasets with keys {'train', 'val', 'test'}.
            - val_loss_func: The loss function used for validation. Either a function or a strin in ['mmr', 'hsic'].
            - test_loss_funcs: A dictionary containing the loss functions used for testing.
            - epsilon: The regularization parameter in the Sinkhorn distance.
            - reg_param: The regularization parameter for the dual model, reg_param= lambda * gamma_t /epsilon.
            - dual_model: The dual model used for training. If not provided, a default architecture will be used.
            - lr_model: The learning rate for the main model.
            - lr_dual: The learning rate for the dual model.
            - batch_size: The batch size used for training.
            - epochs: The number of epochs for training.
            - patience: The number of epochs to wait before early stopping.
            - device: The device used for training (e.g., 'cuda' for GPU or 'cpu' for CPU).
            """
            
            self.model = model
            self.dual_model = dual_model
            self.datasets = datasets
            self.val_loss_func = val_loss_func
            self.loss_funcs = test_loss_funcs if test_loss_funcs else {}
            self.epsilon = epsilon
            self.reg_param = reg_param  # Here `reg_param` corresponds to (\lambda * \gamma_t)/epsilon
            self.dual_model = dual_model
            self.lr_model = lr_model
            self.lr_dual = lr_dual
            self.batch_size = batch_size
            self.epochs = epochs
            self.patience = patience
            self.device = device if torch.cuda.is_available() else 'cpu'

            self.dataloaders = self._get_dataloaders()
            self.loss_funcs["val"] = self._get_val_loss_func()
            self.loss_vals = {mode: defaultdict(list) for mode in self.datasets.keys()}

            sample_data = next(iter(self.dataloaders['train']))

            if self.dual_model is None:
                print("No dual model provided, using default architecture.")
                sample_data = next(iter(self.dataloaders['train']))

                self.dual_model = torch.nn.Sequential(
                torch.nn.Linear(sample_data["z"].shape[1], 50),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(50, 20),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(20, sample_data["y"].shape[1])
            )
            else:
                self.dual_model = dual_model

            self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_model, betas=(0.5, 0.9))
            self.dual_optimizer = torch.optim.Adam(self.dual_model.parameters(), lr=lr_dual, betas=(0.5, 0.9))
            
            self.moment_function = lambda y_pred, y_true: y_pred - y_true

    def _get_dataloaders(self):
        dataloaders = {}

        for mode, data in self.datasets.items():
            if self.batch_size:
                dataloaders[mode] = torch.utils.data.DataLoader(data,
                                                                batch_size=min(self.batch_size, len(data)),
                                                                shuffle=(mode == 'train'),
                                                                drop_last=True,
                                                                num_workers=multiprocessing.cpu_count(),
                                                                pin_memory=True)
            else:
                dataloaders[mode] = torch.utils.data.DataLoader(data,
                                                                batch_size=len(data),
                                                                shuffle=False,
                                                                pin_memory=True)
        return dataloaders

    def _get_val_loss_func(self):
        if self.val_loss_func is None:
            return lambda x, y: torch.Tensor([np.inf])

        if not isinstance(self.val_loss_func, str):
            return self.val_loss_func
        else:
            if self.val_loss_func == 'mmr':
                return mmr
            elif self.val_loss_func == 'hsic':
                return hsic
            else:    
                raise NotImplementedError
        
    def data_jac_per_sample(self, data):
        """
        Consider only the case gamma_z = \infty to reduce complexity
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
        for i in range(data['y'].shape[1]):
            hes = torch.autograd.functional.hessian(lambda t: self.moment_function(self.model(t), data['y']).sum(0)[i], inputs=t, create_graph=True)
            laplacian.append(torch.einsum("ijij -> i", hes))
        laplacian = torch.stack(laplacian, dim=1)
        return laplacian

    def objective(self, data, *args, **kwargs):
        model_pred = self.model(data['t'])
        hz = self.dual_model(data['z'])
        moment_func = self.moment_function(model_pred, data['y'])
        m_vector = (moment_func * hz).sum(1)
        moment = m_vector.mean()

        jac = self.data_jac_per_sample(data)
        jac_sq = torch.einsum("nyt, nzt -> nyz", jac, jac)

        covariance = 0.5 * self.epsilon * torch.einsum("nz, nzy, ny -> ", hz, jac_sq, hz) / len(hz)
        laplace = 0.5 * self.epsilon * self.data_laplacian_per_sample(data).mean()

        if self.reg_param > 0:
            l_reg = self.reg_param * (hz ** 2).mean()
        else:
            l_reg = 0

        primal_loss = moment + laplace
        return primal_loss, -primal_loss + covariance + l_reg
    
    def _early_stopping(self, loss, epochID):
        if self.val_loss_func is None:
            return False
        
        if loss < self.best_val:
            self.best_val = loss
            self.last_best = epochID
        elif self.patience and (epochID - self.last_best > self.patience):
            return True
        return False

    def fit(self):
        self.model = self.model.to(self.device)

        self.best_val = np.inf
        self.last_best = 0

        tqdm_iter = tqdm(range(self.epochs), dynamic_ncols=True)
        stop = False

        for epoch in tqdm_iter:
            for mode in self.dataloaders.keys():
                val_loss = self._epoch(mode=mode)
                if mode == 'val':
                    stop = self._early_stopping(val_loss[-1], epoch)
                    
            if stop:
                print("No improvement in the last {self.patience} epochs. Early stopping ...")
                break

            tqdm_iter.set_description("Obj: {:.4f} | Val Metric: {:.4f}".format(self.loss_vals['train']['objective'][-1],
                                                                                self.loss_vals['val']['val'][-1]), 
                                                                                refresh=True)

        self.is_trained = True

    def _epoch(self, mode):
        train = 'train' in mode
        self.model.train() if train else self.model.eval()

        all_losses = defaultdict(list)

        for batch in self.dataloaders[mode]:
            batch = {key: val.to(self.device) for key, val in batch.items()}

            if train:
                self.dual_optimizer.zero_grad()
                _, dual_objective = self.objective(batch)
                dual_objective.backward()
                self.dual_optimizer.step()

                self.model_optimizer.zero_grad()
                model_objective, _ = self.objective(batch)
                model_objective.backward()
                self.model_optimizer.step()

                all_losses['objective'].append(model_objective.item())
            else:
                y_pred = self.model(batch['t'])
                for name, loss_func in self.loss_funcs.items():
                    all_losses[name].append(loss_func(y_pred, batch).item())
                
        for key, val in all_losses.items():
            self.loss_vals[mode][key].append(np.mean(val))

        return self.loss_vals[mode]['val']



##############################################################################################################
####################### Util functions (redundant in each file to make estimator files self-contained ########

def mmr(y_pred, data):
    kernel_z, _ = get_rbf_kernel(data['z'])
    moment_func_val = y_pred - data['y']
    loss = torch.einsum('ir, ij, jr -> ', moment_func_val, kernel_z, moment_func_val) / (moment_func_val.shape[0] ** 2)
    return loss


def hsic(y_pred, data):
    moment_func_val = y_pred - data['y']
    kernel_z, _ = get_rbf_kernel(data['z'])
    kernel_moment, _ = get_rbf_kernel(moment_func_val, moment_func_val)
    m = kernel_moment.shape[0]   # batch size
    centering_matrix = torch.eye(m) - 1.0/m * torch.ones((m, m))
    centering_matrix = centering_matrix.to(kernel_moment.device)
    hsic_ = torch.trace(torch.mm(kernel_z, torch.mm(centering_matrix, torch.mm(kernel_moment, centering_matrix)))) / ((m-1)**2)
    return hsic_


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





if __name__ == "__main__":
    from example_exp import NetworkIVData
    import matplotlib.pyplot as plt

    train_ds = NetworkIVData(1000, ftype="sin")
    val_ds = NetworkIVData(1000, ftype="sin")
    test_ds = NetworkIVData(1000, ftype="sin")

    all_data = next(iter(torch.utils.data.DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)))
    model = train_ds.get_model_factory()()
    smm = NeuralSMM(model, datasets={"train": train_ds, "val": val_ds, "test": test_ds}, test_loss_funcs=train_ds.get_test_loss_funcs())
    smm.fit()
    train_ds.show_function(model, test_data=all_data, train_data=all_data, title="Neural-SMM")

    test_losses = smm.loss_vals['test']
    fig, ax = plt.subplots(len(test_losses.keys()))
    for i, (key, vals) in enumerate(test_losses.items()):
        ax[i].plot(vals, label=key)
        ax[i].legend()
    plt.show()
