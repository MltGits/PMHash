# 这个模块主要是用来放置网络模型(生成软伪标签模型和哈希生成模型)的

import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
import torchvision
from torch.nn.parameter import Parameter
from torch.nn import Linear

def EuclideanDistance(t1, t2):
    N, C = t1.size()
    M, _ = t2.size()
    dist = -2 * torch.matmul(t1, t2.permute(1, 0))
    dist += torch.sum(t1 ** 2, -1).view(N, 1)
    dist += torch.sum(t2 ** 2, -1).view(1, M)
    dist = torch.sqrt(dist)
    return dist

class my_hash(nn.Module):
    def __init__(self, args, multilabelnet=None):
        super(my_hash, self).__init__()

        self.hash_bit = args.hash_bit

        self.deephashingnet = DeepHashingNet(args)

        laten_feats_dim = args.laten_dim
        # self.laten_feats_layer = nn.Linear(self.hash_bit, laten_feats_dim, bias=False)
        self.laten_feats_layer = nn.Linear(laten_feats_dim,self.hash_bit, bias=False)

        self.multilabelnet = multilabelnet

        self.slow_lr_paramaters = nn.Sequential(self.deephashingnet.features, self.deephashingnet.avgpool,
                                                self.deephashingnet.classifier)

        self.fast_lr_paramaters = nn.Sequential(self.multilabelnet, self.deephashingnet.hash_layer,
                                                self.laten_feats_layer)

    def forward(self, x, x_features=None):
        x_hash = self.deephashingnet(x)

        if x_features is not None:

            # 和哈希码一样大小
            x_laten_c = self.laten_feats_layer(self.multilabelnet.cluster_layer)
            dist = EuclideanDistance(x_hash, x_laten_c)
            sum = torch.sum(dist, dim=1, keepdim=True)
            x_laten_q = torch.div(dist, sum)

            # 这里通过多标签生成网络得到生成的微软标签
            x_reconstruct1, x_target_q = self.multilabelnet(x_features)

            return x_hash, x_reconstruct1, x_laten_q, x_target_q
        else:
            return x_hash


# 以预先训练的alexnet为基础，生成长度为args.hash_bit的类哈希码
class DeepHashingNet(nn.Module):
    def __init__(self, args):
        super(DeepHashingNet, self).__init__()
        self.hash_bit = args.hash_bit

        self.base_model = torchvision.models.alexnet(pretrained=True)

        self.features = self.base_model.features
        self.avgpool = self.base_model.avgpool
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), self.base_model.classifier[i])

        feature_dim = self.base_model.classifier[6].in_features
        self.last_hash_layer = nn.Linear(feature_dim, self.hash_bit)

        self.last_layer = nn.Tanh()

        self.hash_layer = nn.Sequential(self.last_hash_layer, self.last_layer)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x_hash = self.hash_layer(x)
        return x_hash


class AE(nn.Module):
    # enc_dims = [2048, 512, 512, 256, 256]
    def __init__(self, laten_dim, enc_dims=[2048, 1024, 512, 256, 128]):
        super(AE, self).__init__()

        self.laten_dim = laten_dim

        self.encoder = nn.Sequential(
            Linear(enc_dims[0], enc_dims[1]), nn.ReLU(),
            Linear(enc_dims[1], enc_dims[2]), nn.ReLU(),
            Linear(enc_dims[2], enc_dims[3]), nn.ReLU(),
            Linear(enc_dims[3], enc_dims[4]), nn.ReLU(),
            Linear(enc_dims[4], laten_dim),
        )
        self.decoder1 = nn.Sequential(
            Linear(laten_dim, enc_dims[4]), nn.ReLU(),
            Linear(enc_dims[4], enc_dims[3]), nn.ReLU(),
            Linear(enc_dims[3], enc_dims[2]), nn.ReLU(),
            Linear(enc_dims[2], enc_dims[1]), nn.ReLU(),
            Linear(enc_dims[1], enc_dims[0])
        )

    def forward(self, x):

        x_compressed = self.encoder(x)
        x_reconstruct1 = self.decoder1(x_compressed)

        return x_reconstruct1, x_compressed


class MultilabelNet(nn.Module):

    def __init__(self,
                 n_laten_dim,  # 64(经过自动编码器后的输出)
                 n_clusters=10,  # 聚类的数量默认是(10)
                 ae=None):
        super(MultilabelNet, self).__init__()
        self.ae = ae
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_laten_dim))
        self.alpha = 1.0
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x):

        x_reconstruct1, x_compressed = self.ae(x)

        dist = EuclideanDistance(x_compressed, self.cluster_layer)
        sum = torch.sum(dist, dim=1, keepdim=True)
        q = torch.div(dist, sum)


        return x_reconstruct1, q
