"""
ResNetBase网络结构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import argparse
import torch.optim as optim

parse = argparse.ArgumentParser(description="ResNetBase net arguments")
parse.add_argument("--batch-size", default=16, type=int,
                   help="batchsize for the model training")
parse.add_argument("--epoch", default=20, type=int,
                   help="the epoch of the training")
parse.add_argument("--input-dim", default=3, type=int,
                   help="the resnet input dimension")
parse.add_argument("--output-dim", default=10, type=int,
                   help="the resnet output dimsion")
parse.add_argument("--device", default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                   help="device name")


class ResNetBase(nn.Module):
    def __init__(self, _input_dim, _output_dim, args):
        """

        :param _input_dim: input dimension
        :type _input_dim: int
        :param _output_dim: output dimension
        :type _output_dim: int
        """
        super(ResNetBase, self).__init__()
        self.intput_dim = _input_dim
        self.output_dim = _output_dim
        self.conv1 = nn.Conv2d(in_channels=self.intput_dim, out_channels=self.output_dim,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.output_dim)
        self.conv2= nn.Conv2d(in_channels=self.output_dim, out_channels=self.output_dim,
                              kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.output_dim)
        if self.intput_dim != self.output_dim:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=self.intput_dim, out_channels=self.output_dim,
                          kernel_size=1, stride=1),
                nn.BatchNorm2d(self.output_dim))
        self.fc1 = nn.Linear(in_features=self.output_dim*32*32, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, input_x):
        """

        :param intput_x: batchsize * channels * w * h
        :type intput_x: tensor
        """
        batch_size = input_x.size()[0]
        x_channel = input_x.size()[1]
        output = F.relu(self.bn1(self.conv1(input_x)))
        output = self.bn2(self.conv2(output))
        # short cut
        if x_channel != output.size()[1]:
            x = self.layer1(input_x)
        output += x
        output = output.view(batch_size, -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = F.log_softmax(output, dim=0)# 按列求值
        return output


def train(_model, _optimizer, _train_loader, _device, _epoch, args):
    _model.train()
    args.device = _device
    for index, (data, label) in enumerate(_train_loader):
        data = data.to(args.device)
        label = label.to(args.device)
        _optimizer.zero_grad()
        output = _model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        _optimizer.step()
        if index%1000 == 0:
            print(f"epoch is {_epoch}, and train loss is {loss}")



def test(_model, _test_loader, _device, _epoch, args):
    _model.eval()
    args.device = _device
    correct = 0.0
    with torch.no_grad():
        for data, label in _test_loader:
            data = data.to(args.device)
            label = label.to(args.device)
            output = _model(data)
            test_loss = F.cross_entropy(output, label)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss = test_loss/len(_test_loader.dataset)
        correct = correct*100/len(_test_loader.dataset)
        print(f"test_loss is {test_loss}, and correct is {correct}")


def main():
    # print("\033[32;1m{}\033[0m".format("main 函数已进入"))
    args = parse.parse_args()
    print(args)
    pipline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])
    train_datasets = datasets.CIFAR10("D:/Datasetsd/DataSets/CIFAR/CIFAR10", train=True,
                                      download=True, transform=pipline)
    train_loader = DataLoader(dataset=train_datasets, shuffle=True, batch_size=args.batch_size)

    test_datasets = datasets.CIFAR10("D:/Datasetsd/DataSets/CIFAR/CIFAR10", train=False,
                                     download=True, transform=pipline)
    test_loader = DataLoader(dataset=test_datasets, shuffle=True, batch_size=args.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNetBase(args.input_dim, args.output_dim, args).to(device)
    optimizer = optim.Adam(model.parameters())

    epoch = args.epoch

    # train
    for e in range(epoch):
        train(model, optimizer, train_loader, device, e, args)
        test(model, test_loader, device, e, args)



if __name__ == '__main__':
    main()