import torch
import torchvision
from torchvision import datasets, models
import argparse
import detectron2


# models的参数使用
model_names = sorted(model for model in models.__dict__
                    if model.islower() and not model.startswith("__")
                     and callable(models.__dict__[model]))
print(model_names)
# ['alexnet', 'densenet121', 'densenet161', 'densenet169',
# 'densenet201', 'googlenet', 'inception_v3', 'mnasnet0_5',
# 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2',
# 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50',
# 'resnext101_32x8d', 'resnext50_32x4d', 'shufflenet_v2_x0_5',
# 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
# 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
# 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2',
# 'wide_resnet50_2']

print(len(model_names))
#35

new_model = models.__dict__["resnet50"]()# 调用模型类初始化函数
for index, (name, param) in enumerate(new_model.named_parameters()):
    # print(f"This is no.{index} layer, name is {name}, para_shape is {param.shape}")
    if name not in ["fc.weight", "fc.bias"]:
        param.requires_grad = False

print(new_model.fc.weight.shape)
print(new_model.fc.weight.shape[1])

x = [[1, 2],[2, 3], [2, 1]] # 3x2 -> [2, 1, 3]
print("---------------------这是分界线----------------------")
# print("fc.weight is ", new_model.fc.weight.data)
# print("fc.bias is ", new_model.fc.bias.data)
#初始化fc参数
new_model.fc.weight.data.normal_(mean=0.0, std=0.01)
new_model.fc.bias.data.zero_()


# print("afer, fc.weight is ", new_model.fc.weight)
# print("after, fc.bias is ", new_model.fc.bias)

path = "./moco_v2_800ep_pretrain.pth.tar"

checkpoint = torch.load(path, map_location='cpu')
state_dict = checkpoint["state_dict"]
# print(state_dict)
#
# print(checkpoint.keys())

for k in list(state_dict.keys()):
    if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
        state_dict[k[len("module.encoder_q."):]] = state_dict[k]

for k in list(state_dict.keys()):
    print(k)

msg = new_model.load_state_dict(state_dict, strict=False)
print(msg)


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default="./data/",
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8009', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=10, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='./model/moco_v2_800ep_pretrain.pth.tar', type=str,
                    help='path to moco pretrained checkpoint')

args = parser.parse_args()

for key, value in vars(args).items():
    print(key, value)

def func(parm_a, parm_b, parm_c):
    """
    :param parm_a: type parm_a: int
    :param parm_b: type parm_b: int
    :param parm_c: type parm_c: bool
    :return: rtype: int
    """
    return parm_a < parm_b
