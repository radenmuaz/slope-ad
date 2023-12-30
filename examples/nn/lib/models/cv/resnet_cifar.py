import math
import slope
from slope import nn
from functools import partial
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        # cfg should be a number in this case
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x, training=False):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out



def downsample_basic_block(x, planes):
    x = slope.avgpool2d(x, (2, 2))
    # out = x.pad(0,0,0,planes - x.size(1),0,0,0,0)
    zero_pads = slope.zeros(x.size(0), planes - x.size(1), x.size(2), x.size(3))
    out = slope.cat([x, zero_pads], dim=1)
    return out


class ResNet(nn.Module):
    def __init__(self, depth, dataset="cifar10", cfg=None):
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, "depth should be 6n+2"
        n = (depth - 2) // 6

        block = BasicBlock
        if cfg == None:
            cfg = [[16] * n, [32] * n, [64] * n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.cfg = cfg

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[n : 2 * n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[2 * n : 3 * n], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        if dataset == "cifar10":
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes * block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))

        return nn.Sequential(*layers)

    def __call__(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = self.relu(x)  # 32x32

        x = self.layer1(x, training)  # 32x32
        x = self.layer2(x, training)  # 16x16
        x = self.layer3(x, training)  # 8x8

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return (x, self) if training else x


def resnet(**kwargs):
    return ResNet(**kwargs)


if __name__ == "__main__":
    net = resnet(depth=20)
    x = slope.tensor.rand(16, 3, 32, 32)
    y = net(x)
    print(y.data.shape)
