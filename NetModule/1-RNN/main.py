import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.nn.modules.rnn import RNN
import torch.optim as optim
from torch.optim import optimizer
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCH = 50

# 数据预处理
pipline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST("data", train=True, download=True, transform=pipline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipline)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


# 定义一个RNN模型
class RNN_CLASS(nn.Module):
    def __init__(self, _input_dimension, _hidden_dimension, _layer_dimension, _output_dimension):
        # RNN需要四个输入参数
        # _input_dimension 模型的输入的维度
        # _hidden_dimension 模型隐藏层的维度
        # _layer_dimension 模型需要几层RNN
        # _output_dimension 模型的输出的维度
        super(RNN_CLASS, self).__init__()
        self._input_dimension = _input_dimension
        self._hidden_dimension = _hidden_dimension
        self._layer_dimension = _layer_dimension
        self._output_dimension = _output_dimension

        # 定义网络
        # input的默认shape (L, N, Hin) L length of input seq; N batch size; Hin features of the input seq
        self.RNN = nn.RNN(input_size=_input_dimension, hidden_size=_hidden_dimension, num_layers=_layer_dimension,
                          nonlinearity="relu", batch_first=True)
        self.fc = nn.Linear(in_features=_hidden_dimension, out_features=_output_dimension)

    def forward(self, x):
        # 初始化隐藏层的状态
        # tensor在构建计算图的时候，h0是最初的计算图的状态
        # 为了防止梯度爆炸，这里需要将这最初的状态进行分离
        # 原因是：当一个序列中的每一个单元依次被输入到RNN中，如果不进行梯度分离，它会记忆并存储每一次计算的计算图
        # 根据反向传播的公式，这种情况会导致梯度的指数级爆炸
        # 为了不产生梯度爆炸，在进行反向传播的过程中，只考虑seq最后一次的单元输入

        # RNN的输入：input h_0
        # input: tensorshape is (L, N, Hin) if batch_first is False else tensorshape is (N, L Hin) 主要包括了输入序列的特征
        # L length of input seq; N batch size; Hin features of the input seq
        # h_0：tensorshape is (D*num_layers, N, Hout)
        # D = 2 if bidirectional=True otherwise 1; num_layers：hidden层; Hout hidden size
        h0 = torch.zeros(self._layer_dimension, x.size(0), self._hidden_dimension).requires_grad_().to(device=DEVICE)

        # RNN的输出：output, hn
        # output: tensorshape is (L, N, D*Hout) if batch_first is false else tensorflow is s(N, L, D*Hout)
        # L: length of the seq, 这里的seq指的是输入的每一次单元产生的最后一层的输出 均会保存在这个output中，我们通常只需要最后一次的即可
        # hn: tensorshape is (D*num_layers, N, Hout) 这个就是包含了最后的隐藏层的状态
        out, hn = self.RNN(x, h0.detach())
        # print()
        # out, hn = self.RNN(x, h0)
        print(sys.getsizeof(hn))
        # print("the out shape is ", out.shape)
        output = self.fc(out[:, -1, :])
        return output


# 初始化模型
input_dimension = 28
hidden_dimension = 100
layer_dimension = 3
output_dimension = 10  # 输出维度——因为这里的RNN做的是手写数字图像识别，输出应该是是个10个概率 所以设置为10个维度

model = RNN_CLASS(input_dimension, hidden_dimension, layer_dimension, output_dimension).to(DEVICE)

# 定义优化器
my_optimizer = optim.SGD(model.parameters(), lr=0.001)

test_loss_list = []
accuracy_list = []
train_loss_list = []
epoch_list = []


# 定义训练函数
sequence_dimension = 28#序列长度

def train(_model, _optimizer, _train_loader, _device, _epoch):
    model.train()
    for batch_index, (data, label) in enumerate(_train_loader):
        # 一个batch的数据转换为rnn的输入维度
        # print("the orignal datashape is ", data.shape)
        data = data.view(-1, sequence_dimension, input_dimension).requires_grad_().to(DEVICE)
        # print("the datashape after transform is ", data.shape)
        label = label.to(DEVICE)
        _optimizer.zero_grad()
        output = model(data)
        # print("the output shape is ", output.shape)
        loss = F.cross_entropy(output, label)
        loss.backward()
        _optimizer.step()
        if batch_index % 1000 == 0:
            print("The Epoch is :{} \t The train loss is :{:.4f}".format(_epoch, loss.item()))
            train_loss_list.append(loss.item())


# 定义测试函数
def test(_model, _test_loader, _device):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, label in _test_loader:
            data = data.view(-1, sequence_dimension, input_dimension).to(DEVICE)
            label = label.to(DEVICE)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            test_loss += F.cross_entropy(output, label)
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(_test_loader.dataset)
        test_loss_list.append(test_loss)
        accuracy_list.append(100 * correct / len(_test_loader.dataset))
        print("Test——Loss rate is {:.4f}\t Accuracy is{:.3f}".format(test_loss, correct * 100 / len(_test_loader.dataset)))


for e in range(1, EPOCH + 1):
    train(model, my_optimizer, train_loader, DEVICE, e)
    test(model, test_loader, DEVICE)
    epoch_list.append(e)
