import torch
import torchvision
from torchvision import datasets, models
import torch.distributed as dist
import random
import argparse
from torch.backends import cudnn
import os
import torch.multiprocessing as mp

# 通用式分布式常见概念
# group: 进程组。默认情况下只有一个组，一个job就是一个组，也就是一个world，当我们使用多进程的时候，
# 一个group就也有了多个world. 当需要更加精细的通信的时候，可以通过new_group接口，使用world的子集，创建
#组，用于集体通信等
# world：全局进程的个数
# rank：表示进程的序号，用于进程之间的通信，可以用于表示进程的优先级，一般设置rank=0的主机为master节点
# local_rank：进程内GPU编号，非显式参数，由torch.distributed.launch内部指定。
# e.g. rank = 3，local_rank = 0表示第三个进程内的第一块GPU

# argparse 模块
# 用于编写命令行窗口 程序用来定义所需要的参数
# 或者是用来创建一个args的字典

model_names = sorted(name for name in models.__dict__
                     if name.islower() and name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description="Distributed Learning")# 初始化

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
parser.add_argument('--world-size', default=1, type=int,
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
                    default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='./model/moco_v2_800ep_pretrain.pth.tar', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--distributed', default=True,
                    help="judge if start the distributed")
parser.add_argument('--master', default=False,
                    help='tag to define the master process')

# 启动多进程任务
# 获取gpu的个数
ngpus_per_node = torch.cuda.device_count()
# print(f"we have {ngpus_per_node} gpus")

def main_worker(gpu, args):
    # 是每一个进程实际执行的任务
    # 函数中的代码在每一个进程中都会运行一次，不同的进程使用了不同的rank，后续也是通过这个来区分不同的进程
    # 一般选择一个master 也就是rank = 0用于打印信息
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU :{}".format(args.gpu))
    global best_result
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + args.gpu
        # backend GPU选择nccl CPU选择gloo 不用选了
        # init_method参数是多进程的通信的方式，通过dist_url 单机多卡无脑使用tcp
        # tcp://127.0.0.1:8009 随便选择一个没有占用的端口就行
        torch.distributed.init_process_group(backend=args.dist_backend,
                                             init_method=args.dist_url,
                                             world_size=args.world_size,
                                             rank=args.rank)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank%ngpus_per_node == 0):
            args.master = True
        else:
            args.master = False
        # 数据处理部分
        # 以往使用nn.DataParallel接口之所以简单 是因为数据是在全局中进行处理的，所以不需要对DataLoader进行特别的处理
        # Pytorch分布式训练的原理是将数据切成world_size份，然后在每一个进程之中单独进行处理数据、前向和反向传播
        # 所以会快一些 但是也是因此需要对DataLoader进行一些处理


def main():
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("\033[33;1m{}\033[0m".format("gpu is not available, cpu training will be very slow"))
    else:
        print("\033[32;1m{}\033[0m".format("let's use {} gpus").format(ngpus_per_node))

    if args.seed is not None:
        random.seed(args.seed)# 设置随机数生成器的种子 也就是random函数的种子
        torch.manual_seed(args.seed)# 设置CPU生成随机数的种子
        # cudnn中对卷积操作进行了优化 牺牲了精度来换取计算效率，如果需要保证可重复性，可以设置如下
        cudnn.deterministic = True
        print("\033[32;1m{}\033[0m".format("you have choson to seed training, This will turn "
                                           "on the CUDNN deterministic setting, which can slow"
                                           "down your training considerably"))
        args.world_size = 2
        if args.dist_url == "env://" and args.world_size == -1:
            args.word_size = int(os.environ["WORLD_SIZE"])
        print("\033[32;1m{}\033[0m".format("一共进程数目为{}").format(args.world_size))

        args.distributed = args.world_size > 1 or args.multiprocessing_distributed

        if args.multiprocessing_distributed:
            # 我们定义一下每一个gpu运行几个进程 这样需要重新调整一下world_size
            # 为真就可以启动多进程分布式训练
            # 通过spawn直接提交每一个进程的任务
            args.world_size = ngpus_per_node * args.world_size
            # spawn 生成一个子进程 父进程获取子进程的输出
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=args)
        else:
            main_worker(args.gpu, args)

main()