# LetNet网络实现手写数字图像识别
# 数据集 MINIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 参数初始化
EPOCH = 50
BATCH_SIZE = 16
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集下载 数据集处理
pipline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081, ))
])

train_set = datasets.MNIST("data", download=True, train=True, transform=pipline)
test_set = datasets.MNIST("data", download=True, train=False, transform=pipline)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# 模型定义
# 原模型的输入尺寸为32 * 32 这里需要对其进行padding修正
# input batchsize * 1 * 28 * 28
# 卷积层1 kernel_size = 5*5 kernel_num = 6 stride = 1 padding = 0(2)-为了和原模型输入尺寸对应
# output: batchsize * 6 * 28 * 28
# ReLU
# 下采样（平均池化）：2*2 padding = 0
# output: batchsize * 6 * 14 * 14
# 卷积层2 kernel_size = 5*5 kernel_num = 16 stride = 1 padding = 0
# output: batchsize * 16 * 10 * 10
# ReLU
# 下采样（平均池化）：2*2 padding 0
# output:batchsize * 16 * 5 * 5
# FC1：120
# FC2：84
# FC3（output）：10

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # self.input_dimension = _input_dimension
        # self.output_dimension = _output_dimension
        # inputsize batchsize * 1 * 28 * 28
        self.cov1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
                              stride=1, padding=2)
        self.cov2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.outputLayer = nn.Linear(84, 10)
    def forward(self, x):
        input_size = x.size(0)
        x = self.cov1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x,2, 2)
        x = self.cov2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2, 2)
        x = x.view(input_size, -1)#拉平
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.outputLayer(x)
        output = F.log_softmax(x, dim=0)# dim = 0 按照行来计算
        return output

# 优化器初始化
model = LeNet5().to(DEVICE)
optimizer = opt.Adam(model.parameters())

# 画图数据存储
correct_list = []
train_loss_list = []
test_loss_list = []
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