import torch
from torch import nn
import torch.nn.functional as F

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[1]
            b = L2_distances.shape[0]
            L2_distances = L2_distances.reshape(b,-1)
            return torch.sum(L2_distances,dim=-1) / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[:,None, ...] / (self.get_bandwidth(L2_distances)[:,None] * self.bandwidth_multipliers)[:,:,None,None]).sum(dim=1)

class PoliKernel(nn.Module):

    def __init__(self, constant_term=1, degree=2):
        super().__init__()
        self.constant_term = constant_term
        self.degree = degree
    def forward(self, X):
        K = (torch.bmm(X, X.permute(0,2,1)) + self.constant_term) ** self.degree
        return K

class LinearKernel(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, X):
        K = torch.bmm(X, X.permute(0,2,1))
        return K

class LaplaceKernel(nn.Module):

    def __init__(self):
        super().__init__()
        self.gammas = torch.FloatTensor([0.1, 1, 5]).cuda()
    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2 #[8, 4, 4]
        return torch.exp(-L2_distances[:,None,...] * (self.gammas)[:, None, None]).sum(dim=1)


class M3DLoss(nn.Module):

    def __init__(self, kernel_type):
        super().__init__()
        if kernel_type == 'gaussian':
            self.kernel = RBF()
        elif kernel_type == 'linear':
            self.kernel = LinearKernel()
        elif kernel_type == 'polinominal':
            self.kernel = PoliKernel()
        elif kernel_type == 'laplace':
            self.kernel = LaplaceKernel()

    def forward(self, X, Y):
        #print(torch.vstack([X, Y]).shape)
        K = self.kernel(torch.concat([X, Y],dim=1))
        b, X_size= X.shape[0],X.shape[1]
        XX = K[:,:X_size, :X_size].reshape(b,-1).mean(1)
        XY = K[:,:X_size, X_size:].reshape(b,-1).mean(1)
        YY = K[:,X_size:, X_size:].reshape(b,-1).mean(1)
        return (XX - 2 * XY + YY).mean()

# m3dloss = M3DLoss("gaussian")
# a = torch.randn(8,4,512)
# b = torch.randn(8,4,512)
# loss = m3dloss(a,b)
# print(loss)
