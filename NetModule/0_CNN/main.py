import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# MNIST数据集
# 训练数据集有6w张
# 测试数据集有1w张

# 定义超参数
BTACH_SIZE = 64
EPOCH = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建pipeline 对图像进行处理
pipline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081, ))
])

# 下载数据 加载数据
from torch.utils.data import DataLoader

train_set = datasets.MNIST("data", train=True, download=True, transform=pipline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipline)

train_loader = DataLoader(train_set, batch_size=BTACH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BTACH_SIZE, shuffle=True)

# 构建训练网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #图片输入的尺寸：batch_size * channel * w * h
        # batch_size * 1 * 28 * 28
        self.cov1 = nn.Conv2d(1, 20, 5, 1)# arg1：卷积层的channel数 arg2: 卷积层的filter的数目 \
        # arg3:filter的尺寸 arg4 stride的长度
        self.cov2 = nn.Conv2d(20, 50, 3, 1)
        self.fc1 = nn.Linear(5000, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        input_size = x.size(0)
        # batch_size * 1 * 28 * 28 to batch_size * 20 * 24 * 24
        x = self.cov1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)#长宽变为二分之一
        # batch_size * 20 * 12 * 12 to batch_size * 50 * 10 * 10
        x = self.cov2(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2, 2)#batch_size * 50 * 10 * 10 to batch_size * 50 * 5 * 5
        # print(x.size())
        x = x.view(input_size, -1)#拉平
        # print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # print(x.size())
        output = F.log_softmax(x, dim=0)#dim = 1按照行来计算
        return output

# 构建优化器
model = Net().to(device=DEVICE)
optimizer = optim.Adam(model.parameters())

# 存储数据 画图使用
train_loss_list = []
test_loss_list = []
accuracy_list = []
epoch_list = []

# 构建训练函数
def train(_model, _train_loader, _optimizer, _device, _epoch):
    model.train()#训练模型
    for batch_index, (data, label) in enumerate(_train_loader):
        data = data.to(_device)
        label = label.to(_device)
        optimizer.zero_grad()
        output = _model(data)
        # pred = output.max(0, keepdim = True)
        # print(data.size())
        # print(output.size())
        # print(label.size())
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        if batch_index%1000 == 0:
            train_loss_list.append(loss.item())
            print("EPOCH IS :{} \t  Loss is {:.6f}".format(_epoch, loss.item()))
# 构建测试函数
def test(_model, _test_loader, _device):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, label in _test_loader:
            data, label = data.to(_device), label.to(_device )
            # print(label)
            output = model(data)# 64*10
            # print("output is ", output.shape)
            # cross_entropy函数的输入
            # input target
            # input: (N, C) target：(N)
            test_loss += F.cross_entropy(output, label)
            # max参数：axis = 0是求列向量的最值 axis=1是求行向量的最值
            pred = output.max(1, keepdim=True)[1]# 值 索引
            # print("pred is ", pred.shape)
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(_test_loader.dataset)
        test_loss_list.append(test_loss)
        accuracy_list.append(correct * 100/len(_test_loader.dataset))
        print("Test——Loss rate is {:.4f}\t Accuracy is{:.3f}".format(test_loss, correct * 100/len(_test_loader.dataset)))

for e in range(1, EPOCH+1):
    train(model, train_loader, optimizer, DEVICE, e)
    test(model, test_loader, DEVICE)
    epoch_list.append(e)

print(len(epoch_list))
print(len(accuracy_list))
print(len(train_loss_list))
print(len(test_loss_list))

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
plt.plot(epoch_list, accuracy_list, marker = "o", color = "r", linestyle = "-")
plt.savefig("./accuracy.jpg")
plt.show()