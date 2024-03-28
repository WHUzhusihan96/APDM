from easydl import *
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34,
            "resnet50": models.resnet50, "resnet101": models.resnet101}
from typing import List, Dict, Optional, Any, Tuple


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


class ResNetFc(nn.Module):
    def __init__(self, res_name):
        super(ResNetFc, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        # self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = F.relu(x)
        return x

    def output_num(self):
        return self.in_features

    def out_features(self) -> int:
        """The dimension of output features"""
        return self.in_features


class AlexNetFc(nn.Module):
    def __init__(self):
        super(AlexNetFc, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.in_features = 4096
        self.features = model_alexnet.features
        self.avgpool = model_alexnet.avgpool
        self.fc1 = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6, self.in_features), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Dropout(), nn.Linear(self.in_features, self.in_features), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def output_num(self) -> int:
        return self.in_features


class VGGFc(nn.Module):
    def __init__(self):
        super(VGGFc, self).__init__()
        model_VGG = models.vgg16(pretrained=True)
        self.in_features = 4096
        self.features = model_VGG.features
        self.avgpool = model_VGG.avgpool
        self.fc1 = nn.Sequential(nn.Linear(512 * 7 * 7, self.in_features), nn.ReLU(True), nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(self.in_features, self.in_features), nn.ReLU(True), nn.Dropout())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def output_num(self) -> int:
        return self.in_features


class denseFc(nn.Module):
    def __init__(self):
        super(denseFc, self).__init__()
        model_dense = models.densenet121(pretrained=True)
        self.in_features = 4096
        self.features = model_dense.features
        self.in_features = model_dense.classifier.in_features

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        x = torch.flatten(out, 1)
        return x

    def output_num(self) -> int:
        return self.in_features


class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck=256):
        super(CLS, self).__init__()
        self.bottleneck_dim = bottle_neck
        self.bottleneck = nn.Linear(in_dim, self.bottleneck_dim)
        self.fc = nn.Linear(self.bottleneck_dim, out_dim)
    # input_feature, bottleneck_feature, fc_feature, predict_prob

    def forward(self, x):
        x = self.bottleneck(x)
        output = self.fc(x)
        return x, output

    def output_num(self) -> int:
        return self.bottleneck_dim


class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y
