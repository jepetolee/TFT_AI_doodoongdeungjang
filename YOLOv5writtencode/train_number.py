import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.ImageFolder(root='D:/TFT_AI_doodoongdeungjang/fa', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='D:/TFT_AI_doodoongdeungjang/fa', transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset))
'''
classes = ('ahri', 'annie','ashe', 'bard', 'caytlin','darius','echo','erellia','ezreal','fiora','fizz','gankplank',
           'gnar','graves','illaoi','jayce','joy','karma','kashi','kogmo','leona','lucian','lulu','malphite','masteryi',
           'modekaiser', 'neeko', 'nocturne', 'notilus', 'poppy', 'rakan','riven','rumble','shaco','shen','sol','soraka','teemo',
           'thresh','vayne','vi','victor','wukong','wurgot','xayah','xindra','xinzao','yasuo','zanna','zarvan4th','zed','zerath',
           'zhin','zicks','zinx')'''

import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear( 2560, 120)
        self.fc2 = nn.Linear(120, 57)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,  2560)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    net = Net().cuda()
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(5):  # 데이터셋을 수차례 반복합니다.

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data
            inputs, labels =inputs.cuda(),labels.cuda()
            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    PATH = 'image.pt'
    torch.save(net.state_dict(), PATH)
