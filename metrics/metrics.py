import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def dis_2_score(dis, return_numpy=True):
    """
    convert distance to score
    :param dis: distance
    :return: score
    """
    w = torch.linspace(1, 10, 10).to(dis.device)
    w_batch = w.repeat(dis.shape[0], 1)
    score = (dis * w_batch).sum(dim=1)
    if return_numpy:
        return score.cpu().numpy()
    else:
        return score

class ACC(torch.nn.Module):
    def __init__(self):
        super(ACC, self).__init__()

    def forward(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred)

class ACCAVA(torch.nn.Module):
    def __init__(self):
        super(ACCAVA, self).__init__()

    def forward(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        # if score>5, then the prediction is correct, otherwise, it is wrong
        y_pred = np.where(y_pred>5, 1, 0)
        y_true = np.where(y_true>5, 1, 0)
        return accuracy_score(y_true, y_pred)
    
class Pearson(torch.nn.Module):
    def __init__(self):
        super(Pearson, self).__init__()

    def forward(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        return pearsonr(y_true, y_pred)[0]
    
class Spearman(torch.nn.Module):
    def __init__(self):
        super(Spearman, self).__init__()

    def forward(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        return spearmanr(y_true, y_pred)[0]