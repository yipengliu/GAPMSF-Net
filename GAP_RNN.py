import os
import numpy as np
import glob
from scipy import io
import torch
from torch.nn import Module
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from time import time
class down_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down_feature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 20, 5, stride=1, padding=2),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 40, 3, stride=1, padding=1),
            nn.Conv2d(40, 60, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(60, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class res_part(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(res_part, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = x1 + x
        x1 = self.conv2(x)
        x = x1 + x
        x1 = self.conv3(x)
        x = x1 + x
        return x
class up_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_feature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 60, 3, stride=1, padding=1),
            nn.Conv2d(60, 60, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(60, 40, 3, stride=1, padding=1),
            nn.Conv2d(40, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, 20, 3, padding=1),
            nn.Conv2d(20, out_ch, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class forward_rnn(nn.Module):

    def __init__(self):
        super(forward_rnn, self).__init__()
        self.extract_feature1 = down_feature(8, 60)
        self.up_feature1 = up_feature(80, 8)
        self.h_h = nn.Sequential(
            nn.Conv2d(80, 60, 3, padding=1),################30,20
            nn.Conv2d(60, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, 20, 3, padding=1),######################20,10,3
        )
        self.res_part1 = res_part(80, 80)
        #self.res_part2 = res_part(40, 40)

    def forward(self, xt1, h):
        ht = h
        x2 = self.extract_feature1(xt1)
        h = torch.cat([ht, x2], dim=1)
        #print(h.shape)
        h = self.res_part1(h)
        ht = self.h_h(h)
        xt = self.up_feature1(h)

        
        return xt1-xt,ht

class Tensor_AMP_net(Module):
    def __init__(self,layer_num,A,Q):
        super().__init__()
        self.layer_num = layer_num
        self.deblocks = []
        self.steps = []
        self.A = A.cuda()
        self.Q = Q.cuda()
        for n in range(layer_num):
            self.deblocks.append(forward_rnn())
            self.register_parameter("step_" + str(n + 1), nn.Parameter(torch.tensor(0.125),requires_grad=True))
            self.steps.append(eval("self.step_" + str(n + 1)))
        for n,deblock in enumerate(self.deblocks):
            self.add_module("deblock_"+str(n+1),deblock)

    def trans_sampling(self,A,data):
        A_temp= torch.unsqueeze(A,dim=0)
        return A_temp*data

    def select_Q(self,A):
        #print(A)
        A = torch.from_numpy(A)
        A = torch.sum(A,dim=0)
        A[torch.eq(A, 0)] = 1
        Q = torch.unsqueeze(A,dim=0)
        Q = Q.expand([10,-1,-1])
        Q = 1/Q
        return Q

    def forward(self, inputs, output_layers):
        
        y = self.sampling(inputs)
        X = self.trans_sampling(self.Q,y)  # 初始化
        h0 = torch.zeros(X.shape[0], 20, 256, 256).cuda()
        y1 = torch.zeros_like(y)
        start = time()
        for n in range(output_layers):
            yb = self.sampling(X)
            step = self.steps[n]
            deblocker = self.deblocks[n]
            y1 = y1 + (y-yb)
            X = X +step*self.trans_sampling(self.Q,(y1-yb))
            if n==0:
                X,ht = deblocker(X,h0)
            X,ht = deblocker(X,ht)
        end = time()
        return X,end-start


    def sampling(self,inputs): 

        inputs = inputs * torch.unsqueeze(self.A,dim=0)
        outputs = torch.sum(inputs,dim=1)
        return torch.unsqueeze(outputs,dim=1)

    def block1(self,X, y, step):
        outputs = self.trans_sampling(self.A, y-self.sampling(X))
        outputs = step * outputs + X
        return outputs




if __name__ == "__main__":
    pass
