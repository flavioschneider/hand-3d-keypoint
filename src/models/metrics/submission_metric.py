
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchmetrics import Metric

class SubmissionMetric(Metric):
    """ 
        Submission metric class compatible with torchmetrics.
    """
    def __init__(self) -> None:
        super().__init__()
        self.module = PerfMetric()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.sum += self.module(pred, target)
        self.count += 1

    def compute(self): 
        return self.sum.float() / self.count


class PerfMetric(nn.Module):
    """
    This is the loss which is computed on the submission server
    WARNING: This is a very costly metric to compute because of procrustes. It is only
    provided so you know what the exact error metric used by the submission system is.
    You are advised to only use sparingly
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_aligned = procrustes(target, pred)
        err = ((pred_aligned - target) ** 2).sum(-1).sqrt().mean()
        return err


def procrustes(X, Y):
    """
    A batch-wise pytorch implementation of the PMSE metric.
    Computes the affine transformation from Y to X via procrustes.
    Adapted from http://stackoverflow.com/a/18927641/1884420

    Arguments

    X: torch.tensor of shape BS x N x M
    Y: torch.tensor of shape BS x N x M

    where BS: Batch size, N: number of points and M: dim of points

    Returns:
        Rt: Transposed rotation matrix
        s: Scaling factor
        t: Translation factor
        Z: s*matmul(Y,Rt) + t
    """

    X = X.float()
    Y = Y.float() 

    if torch.all(X == 0):
        print("X contains only NaNs. Not computing PMSE.")
        return np.nan, Y
    if torch.all(Y == 0):
        print("Y contains only NaNs. Not computing PMSE.")
        return np.nan, Y

    muX = X.mean(dim=1, keepdim=True)
    muY = Y.mean(dim=1, keepdim=True)
    # Center to mean
    X0 = X - muX
    Y0 = Y - muY
    # Compute frobenius norm
    ssX = (X0 ** 2).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
    ssY = (Y0 ** 2).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
    normX = torch.sqrt(ssX)
    normY = torch.sqrt(ssY)
    # Scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY
    # Compute optimum rotation matrix of Y
    A = torch.matmul(X0.transpose(2, 1), Y0)
    U, s, V = torch.svd(A)
    T = torch.matmul(V, U.transpose(2, 1))
    # Make sure we have a rotation
    detT = torch.det(T)
    V[:, :, -1] *= torch.sign(detT).view(-1, 1)
    s[:, -1] *= torch.sign(detT)
    T = torch.matmul(V, U.transpose(2, 1))

    traceTA = s.sum(dim=1).view(-1, 1, 1)
    #b = traceTA * normX / normY
    Z = normX * traceTA * torch.matmul(Y0, T) + muX

    #c = muX - b * torch.matmul(muY, T)

    #Rt = T.detach()
    #s = b.detach()
    #t = c.detach()
    #return Rt, s, t, Z
    return Z 
