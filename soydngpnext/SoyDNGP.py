import torch
from torch import nn


class CA_attention(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_attention, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class CBAM_attention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM_attention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SE_attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CNN_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride, drop_out=0.3):
        super().__init__()
        if padding != 0:
            padding_mode = 'reflect'
            self.block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, padding_mode=padding_mode,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.Dropout(drop_out),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.Dropout(drop_out),
            )

    def forward(self, x):
        return self.block(x)


class Linear_(nn.Module):
    def __init__(self, length, out_features, drop_out=0.3):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Linear(length, out_features, bias=False)
        )

    def forward(self, x):
        return self.linear(x)


class ReLU_(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.Sequential(
            nn.ReLU(),
        )

    def forward(self, x):
        return self.relu(x)


class Rediual_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride, drop_out):
        super().__init__()
        self.stride = stride
        self.c = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, padding_mode='reflect', stride=self.stride,
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(drop_out),
            nn.Conv2d(out_channel, out_channel, 3, 1, padding_mode='reflect', stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(drop_out),

        )
        if self.stride > 1:
            if padding != 0:
                self.r = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, padding_mode='reflect',
                              stride=self.stride, bias=False),
                    nn.BatchNorm2d(out_channel)
                )
            else:
                self.r = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channel)
                )

    def forward(self, x):
        if self.stride > 1:
            o1 = self.c(x)
            out = self.r(x) + o1
        else:
            out = self.c(x)
        return out


class TransCNN_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride):
        super().__init__()
        if padding != 0:
            padding_mode = 'reflect'
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, padding_mode=padding_mode,
                                   stride=stride, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, bias=False),
            )

    def forward(self, x):
        return self.block(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)
