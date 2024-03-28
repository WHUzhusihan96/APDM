from easydl import *
import torch
from torch import nn
import torch.nn.functional as F
import random


def seed_everything(seed=1111):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AccuracyCounterA:

    def __init__(self, length):
        self.Ncorrect = np.zeros(length)
        self.Ntotal = np.zeros(length)
        self.length = length

    def add_correct(self, index, amount=1):
        self.Ncorrect[index] += amount

    def add_total(self, index, amount=1):
        self.Ntotal[index] += amount

    def clear_zero(self):
        i = np.where(self.Ntotal == 0)
        self.Ncorrect = np.delete(self.Ncorrect, i)
        self.Ntotal = np.delete(self.Ntotal, i)

    def each_accuracy(self):
        self.clear_zero()
        return self.Ncorrect / self.Ntotal

    def mean_accuracy(self):
        self.clear_zero()
        return np.mean(self.Ncorrect / self.Ntotal)

    def overall_accuracy(self):
        self.clear_zero()
        return np.sum(self.Ncorrect) / np.sum(self.Ntotal)

    def h_score(self):
        self.clear_zero()
        common_acc = np.mean(self.Ncorrect[0:-1] / self.Ntotal[0:-1])
        open_acc = self.Ncorrect[-1] / self.Ntotal[-1]
        return 2 * common_acc * open_acc / (common_acc + open_acc)


class Inter_pcd(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(Inter_pcd, self).__init__()
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=20000))
        self.classifier = classifier
        self.temp = 10.0

    @staticmethod
    def st_sc(y_s: torch.Tensor, y_t: torch.Tensor, label_s: torch.Tensor, temp) -> torch.Tensor:
        soft_yt1 = F.softmax(y_t, dim=1)
        # soft_ys, soft_yt = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        max_prob_yt, label_t = soft_yt1.max(1)
        soft_ys, soft_yt = F.softmax(y_s / temp, dim=1), F.softmax(y_t / temp, dim=1)
        soft_yt = soft_yt[max_prob_yt > 0.8, :]
        label_t = label_t[max_prob_yt > 0.8]
        loss = torch.tensor([0.0]).cuda()
        count = 0
        for i in range(15):
            index_a = torch.where(label_s == i)
            index_b = torch.where(label_t == i)
            if len(index_a[0]) > 0 and len(index_b[0]) > 0:
                count += (len(index_a[0]) * len(index_b[0]))
                inter_a = soft_ys[index_a[0], :]
                inter_b = soft_yt[index_b[0], :]
                sim_matrix = 1 - F.cosine_similarity(inter_a.unsqueeze(1), inter_b.unsqueeze(0), dim=-1)
                loss += torch.sum(sim_matrix)
        if count == 0:
            return loss
        else:
            return loss / count

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor, label_s: torch.Tensor) -> torch.Tensor:
        fs_grl = self.grl(f_s)
        _, y_s = self.classifier(fs_grl)
        ft_grl = self.grl(f_t)
        _, y_t = self.classifier(ft_grl)
        st_loss = self.st_sc(y_s, y_t, label_s, self.temp)
        loss = (self.temp ** 2) * st_loss
        return loss


def Intra_pcd(y: torch.Tensor, label: torch.Tensor, temp=10.0) -> torch.Tensor:
    soft_y = F.softmax(y / temp, dim=1)
    loss = torch.tensor([0.0]).cuda()
    count = 0
    for i in range(15):
        index = torch.where(label == i)
        if len(index[0]) > 0:
            count += ((len(index[0]) * (len(index[0]) - 1)) / 2)
            class_intra = soft_y[index[0], :]
            sim_matrix = (1 - F.cosine_similarity(class_intra.unsqueeze(1), class_intra.unsqueeze(0), dim=-1))
            loss += (torch.sum(sim_matrix) / 2.0)
    if count == 0:
        return (temp ** 2) * loss
    else:
        return (temp ** 2) * loss / count
