import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, bn=False, relu=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Residual, self).__init__()
        self.skip_layer = Conv(in_channel, out_channel, 1, relu=False)
        if in_channel == out_channel:
            self.need_skip = False
        else:
            self.need_skip = True

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv1 = Conv(in_channel, out_channel // 2, 1, bn=True, relu=True)
        self.conv2 = Conv(out_channel // 2, out_channel // 2, 3, bn=True, relu=True)
        self.conv3 = Conv(out_channel // 2, out_channel, 1)

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return residual + out


class Hourglass(nn.Module):
    def __init__(self, layer, channel, inc=0):
        super(Hourglass, self).__init__()
        nf = channel + inc
        self.res = Residual(channel, channel)
        self.pool = nn.MaxPool2d(2, 2)
        self.res1 = Residual(channel, nf)
        if layer > 1:
            self.hourclass = Hourglass(layer - 1, nf)
        else:
            self.hourclass = Residual(nf, nf)
        self.res2 = Residual(nf, channel)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        res = self.res(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.hourclass(x)
        x = self.res2(x)
        x = self.up(x)
        return res + x


if __name__ == '__main__':
    model = Hourglass(2, 8)
    x = torch.randn((1, 8, 16, 16))
    out = model(x)
    print(out.shape)
