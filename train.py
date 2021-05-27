import torch
import torch.nn as nn
import numpy as np

class Deconfounder_loss(nn.Module):
    def __init__(self, alpha):
        super(Deconfounder_loss, self).__init__()
        self.alpha = alpha
        self.ce_criterion = nn.BCELoss()

    def forward(self,features_div,features_div_relation,conv_diverse_norm,orthogonal,loss_reconstruct):
        loss = {}

        features_div_mean = torch.mean(features_div,2)
        features_div_mean = torch.unsqueeze(features_div_mean,2)
        features_div = (features_div - features_div_mean)
        loss_div = -torch.mean(torch.pow(features_div,2))

        loss_relation = torch.mean(torch.pow(features_div_relation,2))

        loss_total = loss_div + 0.1*loss_relation + 50*orthogonal + 0.001*loss_reconstruct

        loss["loss_div"] = loss_div
        loss["loss_relation"] = 0.1*loss_relation
        loss["loss_orthogonal"] = 50*orthogonal
        loss["loss_reconstruct"] = 0.001*loss_reconstruct
        loss["loss_total"] = loss_total

        return loss_total, loss

def train(net, train_loader, loader_iter, optimizer, criterion, logger, step):
    net.train()
    try:
        _data, _label, _, _, _ = next(loader_iter)
    except:
        loader_iter = iter(train_loader)
        _data, _label, _, _, _ = next(loader_iter)

    _data = _data.cuda()
    _label = _label.cuda()

    optimizer.zero_grad()

    features_div,features_div_relation,conv_diverse_norm,orthogonal,loss_reconstruct = net(_data)

    cost, loss = criterion(features_div,features_div_relation,conv_diverse_norm,orthogonal,loss_reconstruct)

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        logger.log_value(key, loss[key].cpu().item(), step)

    return loss
