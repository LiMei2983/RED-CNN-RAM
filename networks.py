import torch.nn as nn
import torch
import torch.nn.functional as F
from cbam import CBAM
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)
class inception_module(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(inception_module, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
        self.conv2d = nn.Conv2d(225, 1, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs =torch.cat(outputs, 1)
        outputs = self.conv2d(outputs)
        return outputs

class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv_first = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t_last = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.inception_module1 = inception_module(in_channels=96, pool_features=1)
        self.CBAM1 = CBAM(gate_channels=10)
        self.inception_module2 = inception_module(in_channels=1, pool_features=1)
        self.CBAM2 = CBAM(gate_channels=10)
        self.inception_module3 = inception_module(in_channels=1, pool_features=1)
        self.CBAM3 = CBAM(gate_channels=10)
        self.inception_module4 = inception_module(in_channels=1, pool_features=1)
        self.CBAM4 = CBAM(gate_channels=10)
        self.inception_module5 = inception_module(in_channels=1, pool_features=1)
        self.CBAM5 = CBAM(gate_channels=10)

    def forward(self, x):
        # encoder
        residual_1 = x.clone()
        out = self.relu(self.conv_first(x))
        out = self.relu(self.conv(out))
        residual_2 = self.inception_module1(out)
        residual_2 = self.CBAM1(residual_2)
        residual_2 = self.inception_module2(residual_2)
        residual_2 = self.CBAM2(residual_2)
        residual_2 = self.inception_module3(residual_2)
        residual_2 = self.CBAM3(residual_2)
        residual_2 = self.inception_module4(residual_2)
        residual_2 = self.CBAM4(residual_2)
        residual_2 = self.inception_module5(residual_2)
        residual_2 = self.CBAM5(residual_2)
        out = self.relu(self.conv(out))
        out = self.relu(self.conv(out))
        residual_3 = self.inception_module1(out)
        residual_3 = self.CBAM1(residual_3)
        residual_3 = self.inception_module2(residual_3)
        residual_3 = self.CBAM2(residual_3)
        residual_3 = self.inception_module3(residual_3)
        residual_3 = self.CBAM3(residual_3)
        residual_3 = self.inception_module4(residual_3)
        residual_3 = self.CBAM4(residual_3)
        residual_3 = self.inception_module5(residual_3)
        residual_3 = self.CBAM5(residual_3)
        out = self.relu(self.conv(out))

        # decoder
        out = self.conv_t(out)
        out += residual_3
        out = self.conv_t(self.relu(out))
        out = self.conv_t(self.relu(out))
        out += residual_2
        out = self.conv_t(self.relu(out))
        out = self.conv_t_last(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out
