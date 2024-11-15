import torch
import torch.nn.functional as F

import math

import torch.nn as nn
from torch.hub import load_state_dict_from_url

BatchNorm2d = nn.BatchNorm2d


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        # ----------------------------------------------------#
        #   利用1x1卷积根据输入进来的通道数进行通道数上升
        # ----------------------------------------------------#
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # ----------------------------------------------------#
                #   利用深度可分离卷积进行特征提取
                # ----------------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # ----------------------------------------------------#
                #   利用1x1的卷积进行通道数的下降
                # ----------------------------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # ----------------------------------------------------#
                #   利用1x1卷积根据输入进来的通道数进行通道数上升
                # ----------------------------------------------------#
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # ----------------------------------------------------#
                #   利用深度可分离卷积进行特征提取
                # ----------------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # ----------------------------------------------------#
                #   利用1x1的卷积进行通道数的下降
                # ----------------------------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual

        print("---------------")
        width_mult = 0.5

        input_channel = 32  # 32
        last_channel = 1280

        interverted_residual_setting = [
            # t, c, n, s
            # 237,237,32 -> 237,237,16
            [1, 16, 1, 1],
            # 237,237,16 -> 119,119,24
            [6, 24, 2, 2],
            # 119,119,24 -> 60,60,32
            [6, 32, 3, 2],
            # 60,60,32 -> 30,30,64
            [6, 64, 4, 2],
            # 30,30,64 -> 30,30,96
            [6, 96, 3, 1],
            # 30,30,96 -> 15,15,160
            [6, 160, 3, 2],
            # 15,15,160 -> 15,15,320
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0
        # 473,473,3 -> 237,237,32
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]

        # 根据上述列表进行循环，构建mobilenetv2的结构
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # mobilenetv2结构的收尾工作
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        # 最后的分类部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(n_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            'https://github.com/bubbliiiing/pspnet-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
            "./model_data"), strict=False)
    return model


class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        # -----------------------------------------------------#
        #   分区域进行平均池化
        #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        # -----------------------------------------------------#
        self.stages = nn.ModuleList(
            [self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])

        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1,
                      bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend(
            [F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class self_net(nn.Module):
    def __init__(self, num_classes=4, downsample_factor=8, backbone="mobilenet", pretrained=False, aux_branch=True):
        super(self_net, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone == "resnet50":
           aaa=1
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   f4为辅助分支    [30,30,96]
            #   o为主干部分     [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor, pretrained)
            aux_channel = 48  # 96
            out_channel = 160  # 320
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

        # --------------------------------------------------------------#
        #	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   30,30,320 -> 30,30,80 -> 30,30,21
        # --------------------------------------------------------------#
        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(out_channel // 4, num_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch

        if self.aux_branch:
            # ---------------------------------------------------#
            #	利用特征获得预测结果
            #   30, 30, 96 -> 30, 30, 40 -> 30, 30, 21
            # ---------------------------------------------------#
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel // 8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel // 8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel // 8, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.master_branch)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_aux, x = self.backbone(x)
        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()
