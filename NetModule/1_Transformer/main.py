"""
构建Transformer实现CIFAR10的图像分类实现
transformer的构建 子结构
encoder 图像embedding模块
Embedding模块
    将图像当成句子进行编码 编码成seq<vec>的格式
Encoder: 是由多个Block模块构成
    Block模块结构：是由一个Self-Attention模块和一个前馈神经网络模块构成
其实还应该有一个decoder的模块 但是对于图像来说，decoder并没有encoder重要
这里做分类直接用transformer的输出的vec来作为logit了
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import argparse
import copy

parse = argparse.ArgumentParser("This is the args of Transformers")
parse.add_argument("--batch-size", default=16, type=int,
                   help="batch_size")
parse.add_argument("--epoch", default=20, type=int,
                   help="epoch")
parse.add_argument("--device", default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parse.add_argument("--image-size", default=224, type=int,
                   help="image_size of input image")
parse.add_argument("--patche-size", default=16, type=int,
                   help="patches of the images")
parse.add_argument("--hidden-len", default=768, type=int,
                   help="the length of the convert image seq")
parse.add_argument("--dropout-rate", default=0.1, type=float,
                   help="the dropout rate")
parse.add_argument("--input-channels", default=3, type=int,
                   help="the image input channels")
parse.add_argument("--num-attention-head", default=1, type=int,
                   help="the attention head that we need")
parse.add_argument("--num-block-layer", default=1, type=int,
                   help="the number of blocks")
parse.add_argument("--num-decoder-fc1-outdim", default=512, type=int,
                   help="this is the num of the decoder fc1 layer's output dimension")
parse.add_argument("--cls-dim", default=10, type=int,
                   help="thie is the dimension of the cls dim")


class ImageEmbedding(nn.Module):
    """
    构建ImageEmbedding模块
    经过模块之后返回的是一个[bs, hidden_len, patch_num**2]
    主要作用就是构建一个图像的编码器，将图像转换为一个seq<vec>
    本项目中 每一个图像的每一个vector的维度为196
    """
    def __init__(self, args):
        super(ImageEmbedding, self).__init__()
        self.image_size = args.image_size
        self.patch_size = args.patche_size
        self.hidden_len = args.hidden_len# 图像转换成seq之后的len的长度
        self.in_channels = args.input_channels
        self.patch_num = self.image_size / self.patch_size
        n_patches = self.patch_num * self.patch_num#将图片分成多少个patch
        # 在这里需要两个 一个是将图片转换成vector的patch编码
        # 一个是图片的patch的embedding编码
        # 此外，类似于BERT编码的方式，在输入embedding之前，会有一个flag标记，这个标记是用来表示类别的
        # 这个类别flag的维度和正常的vector一样
        self.elayer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_len,
                                kernel_size=self.patch_size, stride=self.patch_size)
        # 这里的表示 用hidden_len来表述转换后的len的长度
        # 每一个kernel获取地feature map就是一个单词
        # 需要对应的position_embedding就是将feature map 拉长以后得到的每一个patch对应的位置
        # 得到的metrix为 bs x hidden_len x patch_num x patch_num
        # 这里的+1 就是对应多；类别
        self.position_embedding = nn.Parameter(torch.zeros(1, self.hidden_len+1, int(self.patch_num*self.patch_num)))
        self.cls_token_func = nn.Parameter(torch.zeros(1, 1, int(self.patch_num*self.patch_num)))
        self.dropout = nn.Dropout(p=args.dropout_rate)

    def forward(self, x):
        bs = x.size()[0]
        cls_token = self.cls_token_func.expand(bs, -1, -1)# bs x 1 x 196
        pos_embedding = self.position_embedding.expand(bs, -1, -1) # bs x (hidden_len+1) x 196
        x = self.elayer(x) # bs x hidden_len x patch_num x patch_num
        x = x.view(bs, self.hidden_len, -1)# bs x hidden_len x 196
        x = torch.cat((x, cls_token), dim=1)# bs x (hidden+len+1) x 196
        embedding  = x + pos_embedding
        embedding = self.dropout(embedding)
        return embedding

# Attention机制的构建
# 需要q query k keys v value
class Attention(nn.Module):
    """
    构建self-attention 机制
    将输入的seq<vec> 转换为 attention_context
    output : attention_context, attention_scores
    shape : [bs, 769, 196]
    """
    def __init__(self, args):
        super(Attention, self).__init__()
        self.image_size = args.image_size
        self.patch_size = args.patche_size
        self.hidden_len = args.hidden_len  # 图像转换成seq之后的len的长度
        self.patch_num = self.image_size / self.patch_size
        self.input_dim = int(self.patch_num * self.patch_num)# vector 作为一个input的时候的输入维度
        self.query = nn.Linear(in_features=self.input_dim, out_features=self.input_dim, bias=False)
        self.key = nn.Linear(in_features=self.input_dim, out_features=self.input_dim, bias=False)
        self.value = nn.Linear(in_features=self.input_dim, out_features=self.input_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        # self.out = nn.Linear(in_features=self.input_dim, out_features=self.input_dim)

    def forward(self, x):
        """

        :param x: input x is seq<vec>
        :type x: [bs, hidden_len+1, 196]
        :return:
        :rtype:
        """
        q = self.query(x)# q bs x 769 x 196
        k = self.key(x)# k bs x 769 x 196
        v = self.value(x)# v bs x 769 x 196
        q_t = torch.transpose(q, dim0=-1, dim1=-2)
        k_t = torch.transpose(k, dim0=-1, dim1=-2)
        attention_scores  = torch.matmul(q, k_t)# bs x 769 x 769
        attention_scores = self.softmax(attention_scores)
        context = torch.matmul(attention_scores, v)# bs x 769 x 196
        # context = self.out(context)
        return context, attention_scores

# 前向传播模块
class Mlp(nn.Module):
    def __init__(self, args):
        super(Mlp, self).__init__()
        self.image_size = args.image_size
        self.patch_size = args.patche_size
        self.hidden_len = args.hidden_len# 图像转换成seq之后的len的长度
        self.in_channels = args.input_channels
        self.patch_num = self.image_size / self.patch_size
        self.vec_len = int(self.patch_num * self.patch_num)
        self.fc1 = nn.Linear(in_features=self.vec_len, out_features=self.vec_len)
        self.fc2 = nn.Linear(in_features=self.vec_len, out_features=self.vec_len)
        self.dropout = nn.Dropout(p=args.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, args):
        super(Block, self).__init__()
        self.image_size = args.image_size
        self.patch_size = args.patche_size
        self.hidden_len = args.hidden_len  # 图像转换成seq之后的len的长度
        self.in_channels = args.input_channels
        self.patch_num = self.image_size / self.patch_size
        self.vec_len = int(self.patch_num * self.patch_num)

        self.attention_module = Attention(args)
        self.attention_layer_norm = nn.LayerNorm(self.vec_len, eps=1e-6)
        self.mlp_module = Mlp(args)
        self.mlp_layer_norm = nn.LayerNorm(self.vec_len, eps=1e-6)

    def forward(self, x):
        res = x
        x, w = self.attention_module(x)
        x += res
        x = self.attention_layer_norm(x)
        res = x
        x = self.mlp_module(x)
        x += res
        output = self.mlp_layer_norm(x)
        return output, w

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.layerlist = nn.ModuleList([Block(args)])
        self.image_size = args.image_size
        self.patch_size = args.patche_size
        self.hidden_len = args.hidden_len  # 图像转换成seq之后的len的长度
        self.in_channels = args.input_channels
        self.patch_num = self.image_size / self.patch_size
        self.vec_len = int(self.patch_num * self.patch_num)

        for _ in range(args.num_block_layer):
            layer = Block(args)
            self.layerlist.append(copy.deepcopy(layer))# >

    def forward(self, x):
        attn_weight = []
        for layer in self.layerlist:
            x, weight = layer(x)
            attn_weight.append(weight)
        return x, attn_weight

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.image_size = args.image_size
        self.patch_size = args.patche_size
        self.hidden_len = args.hidden_len  # 图像转换成seq之后的len的长度
        self.in_channels = args.input_channels
        self.patch_num = self.image_size / self.patch_size
        self.vec_len = int(self.patch_num * self.patch_num)

        self.fc1 = nn.Linear(in_features=self.vec_len, out_features=args.num_decoder_fc1_outdim)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=args.num_decoder_fc1_outdim, out_features=args.cls_dim)

    def forward(self, x):
        cls_x = x[:, 0, :]
        x = self.fc1(cls_x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)# bs x
        output = F.log_softmax(x, dim=0)
        return output

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.image_size = args.image_size
        self.patch_size = args.patche_size
        self.hidden_len = args.hidden_len  # 图像转换成seq之后的len的长度
        self.in_channels = args.input_channels
        self.patch_num = self.image_size / self.patch_size
        self.vec_len = int(self.patch_num * self.patch_num)
        self.embedding = ImageEmbedding(args)
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x):
        x = self.embedding(x)
        x, w = self.encoder(x)
        output = self.decoder(x)
        return output

def main():
    args = parse.parse_args()

    pipline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.image_size, args.image_size)
    ])
    train_datasets = datasets.CIFAR10("D:/Datasetsd/DataSets//CIFAR/CIFAR10", download=True, train=True, transform=pipline)
    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=args.batch_size)

    test_datasets = datasets.CIFAR10("D:/Datasetsd/DataSets/CIFAR/CIFAR10", download=True, train=False, transform=pipline)
    test_loader = DataLoader(test_datasets, shuffle=True, batch_size=args.batch_size)


def test_main():
    args = parse.parse_args()
    img = torch.randn(2, 3, 224, 224)# bs x c x w x h
    # embedding = ImageEmbedding(args)
    # output_embedding = embedding(img)
    # print("the img_embedding is ", output_embedding)
    # print("the img_embedding shape is ", output_embedding.shape)
    # attention = Attention(args)
    # output_sa, _ = attention(output_embedding)
    # print("output_selfattention is{}, and the shape is {}".format(output_sa, output_sa.shape))
    # block = Block(args)
    # output,_ = block(output_embedding)
    # print(output.shape)
    # encoder = Encoder(args)
    # output, _ = encoder(output_embedding)
    # print(output.shape)
    # decoder =Decoder(args)
    # output = decoder(output)
    # print(f"the shape is {output.shape}, and the output is {output}")
    model = Transformer(args)
    output = model(img)
    print(f"the output is {output}, and the output shape is {output.shape}")


if __name__ == '__main__':
    test_main()
