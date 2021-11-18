# AlexNet实现MNIST数据集
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as opt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 参数设置
BATCH_SIZE = 16
EPOCH = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 数据下载 数据处理
pipline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081, ))
])
train_set = datasets.MNIST("data", download=True, transform=pipline)
test_set = datasets.MNIST("data", download=True, transform=pipline)

train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=BATCH_SIZE)
# 模型定义
# Alex 模型(修改参数后的模型)
# input：batch*1*28*28
# Cov1：
# kernel：num 32 size 3 stride 1 padding 1
# output: batch*32*28*28
# relu
# maxpool: size 2 stride 2
# ouput: batch*32*14*14
# LRN: size 3
# Cov2:
# kernel：num 64 size 3 padding 1 stride 1
# output batch*64*14*14
# relu
# maxpool size 2 stride 2
# output batch * 64 * 7 * 7
# LRN:size 3
# Cov3:
# kernel：num 128 size 3 padding 1
# output batch*128*7*7
# relu
# Cov4:
# kernel: num 256 size 3 padding 1
# output batch*256*7*7
# relu
# Cov5；
# kernel：num 256 size 3 padding 1
# output batch * 256 *7*7
# relu
# maxpool size 3 stride 2
# output batch * 256 * 3 * 3
# FC1: 256*3*3 1024
# FC2: 1024 512
# FC3: 512 10
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            nn.LocalResponseNorm(size=3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2),
            # nn.MaxPool2d(stride=1, kernel_size=2),
            nn.LocalResponseNorm(size=3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,stride=1),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc1 = nn.Linear(256*3*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
    def forward(self, x):
        input_size = x.size(0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(input_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output

# 优化器设置
model = AlexNet().to(DEVICE)
optimizer = opt.Adam(model.parameters())

# 数据列表初始化
train_loss_list = []
test_loss_list = []
correct_list = []
epoch_list = []
# 训练
def train(_model, _optimizer, _train_loader, _epoch):
    _model.train()
    for batch_index, (data, label) in enumerate(_train_loader):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        _optimizer.zero_grad()
        output = _model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        if batch_index%3750 == 0:
            train_loss_list.append(loss.item())
            print("EPOCH IS :{} \t  Loss is {:.6f}".format(_epoch, loss.item()))
# 测试
def test(_model, _test_loader):
    _model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, label in _test_loader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            output = _model(data)
            test_loss += F.cross_entropy(output, label)
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(_test_loader.dataset)
        test_loss_list.append(test_loss)
        correct_list.append(100*correct/len(_test_loader.dataset))
        print("Test——Loss rate is {:.4f}\t Accuracy is{:.3f}".format(test_loss,
                                                                     correct * 100 / len(_test_loader.dataset)))

for e in range(1, EPOCH+1):
    train(model, optimizer, train_loader, e)
    test(model, test_loader)
    epoch_list.append(e)

plt.figure(1)
plt.title("train loss")
plt.plot(epoch_list, train_loss_list, marker = "o", color = "r", linestyle = "-")
plt.savefig("./train_loss.jpg")
plt.show()

plt.figure(2)
plt.title("test loss")
plt.plot(epoch_list, test_loss_list, marker = "o", color = "r", linestyle = "-")
plt.savefig("./test_loss.jpg")
plt.show()

plt.figure(3)
plt.title("accuracy")
plt.plot(epoch_list, correct_list, marker = "o", color = "r", linestyle = "-")
plt.savefig("./accuracy.jpg")
plt.show()