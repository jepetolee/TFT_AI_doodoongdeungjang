import torch as T
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import os

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class RCA(nn.Module):
    def __init__(self):
        super(RCA, self).__init__()
        self.inputs = 28
        self.data = 1  # 조건 데이터
        self.lr = 0.01
        self.fc1_dims = 7168  # 7*4개의 데이터를 확대(16X16)
        self.fc2_dims = 1792  # 7*4개의 데이터를 확대 (8x8)
        self.fc3_dims = 448  # 7*4개의 데이터를 확대 (4x4)
        self.fc4_dims = 112  # 7*4개의 데이터를 확대 (2x2)
        self.batches = 28

        self.extender = nn.Sequential(
            nn.Conv2d(in_channels=self.data, out_channels=self.inputs, kernel_size=5, stride=2, padding=2),
            nn.ReLU())

        self.conv1 = Conv(self.inputs, self.fc1_dims, k=7, s=5)
        self.pool1 = BottleneckCSP(self.fc1_dims, self.fc2_dims)
        self.conv2 = Conv(self.fc2_dims, self.fc3_dims, k=5, s=3)
        self.pool2 = BottleneckCSP(self.fc3_dims, self.fc4_dims)
        self.dense = nn.Linear(self.fc4_dims, self.batches)
        self.model = nn.Sequential(self.conv1, self.pool1, self.conv2, self.pool2, self.dense)
        self.Device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.Device)

    def forward(self, batches, data):
        x1 = self.extender(data)
        x2 = T.tensor(batches)
        x = T.mm(x1, x2)
        x = nn.ReLU(self.model(x))
        return f.softmax(x,dim=1)



model = RCA()
optimizer=optim.Adam(model.parameters(),lr=0.01,betas=(0.5,0.99))
criterion =nn.BCELoss()
model_loss= criterion(result,target)#목표와 타겟을 기준으로
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

T.save(model.state_dict(), './batch.pt')


