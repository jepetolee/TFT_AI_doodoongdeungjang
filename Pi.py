import torch as T
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


class RCA(nn.Module):
    def __init__(self):
        super(RCA, self).__init__()
        self.inputs = 28
        self.data = 1  # 조건 데이터
        self.lr = 0.01
        self.fc1_dims = 7168  # 7*4개의 데이터를 확대(16X16)
        self.fc2_dims = 1792  # 7*4개의 데이터를 확대 (8x8)
        self.fc3_dims = 448  # 7*4개의 데이터를 확대 (4x4)
        self.batches = 28

        self.extender = nn.Sequential(
            nn.Linear(1,28),
            nn.ReLU())

        self.conv1 = nn.Linear(self.inputs, self.fc1_dims)
        self.pool1 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.conv2 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.dense = nn.Linear(self.fc3_dims, self.batches)
        self.model = nn.Sequential(self.conv1, self.pool1, self.conv2,  self.dense)
        self.Device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.Device)

    def forward(self, data):
        x = self.extender(data)
        x = self.model(x)
        return x


model = RCA()
T.save(model.state_dict(), './batch.pt')
