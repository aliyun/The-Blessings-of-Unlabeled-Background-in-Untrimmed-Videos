import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
import math


project_num = 5

class Deconfounder(nn.Module):
    def __init__(self, len_feature, num_classes, num_segments):
        super(Deconfounder, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes


        self.conv_diverse_weight = nn.Parameter(torch.randn(project_num,len_feature,1))
        nn.init.kaiming_uniform_(self.conv_diverse_weight, a=math.sqrt(5))

        kernel = [-1,1]
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.kernel = torch.repeat_interleave(self.kernel,repeats=project_num,dim=0)
        self.kernel = nn.Parameter(data = self.kernel,requires_grad=False)


    def forward(self, x):
        batch_size = x.shape[0]

        x_permute = x.permute(0, 2, 1)
        features_div = F.conv1d(x_permute,self.conv_diverse_weight/\
                torch.norm(self.conv_diverse_weight,dim=1,keepdim=True),padding=0)
        features_div_relation = F.conv1d(features_div,self.kernel,groups=project_num)

        conv_diverse_norm = torch.norm(self.conv_diverse_weight)

        projectors = torch.squeeze(self.conv_diverse_weight/\
                torch.norm(self.conv_diverse_weight,dim=1,keepdim=True))
        if project_num>1:
            orthogonal = torch.matmul(projectors,torch.transpose(projectors,1,0)) - torch.eye(project_num).cuda()
            orthogonal = torch.sum(torch.pow(orthogonal,2))
        else:
            orthogonal = torch.sum(projectors-projectors)

        features_div_T = torch.transpose(features_div,2,1)

        feature_reconst = torch.matmul(features_div_T,torch.squeeze(self.conv_diverse_weight/\
                torch.norm(self.conv_diverse_weight,dim=1,keepdim=True)))
        loss_reconst = torch.sum((feature_reconst - x) * (feature_reconst - x))
        loss_reconst = loss_reconst/(feature_reconst.shape[0]*feature_reconst.shape[1])

        return features_div,features_div_relation,conv_diverse_norm,orthogonal,loss_reconst


