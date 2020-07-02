import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Net', 'init_weights']


class Down(nn.Module):
    def __init__(self, size):
        super(Down, self).__init__()
        assert(size > 2)
        size2 = 1 << size
        size2m1 = 1 << (size - 1)

        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(size2m1, size2, kernel_size=3, padding=1),
            nn.BatchNorm2d(size2),
            nn.ReLU(),
            nn.Conv2d(size2, size2, kernel_size=3, padding=1),
            nn.BatchNorm2d(size2),
            nn.ReLU())

    def forward(self, X):
        # print('Xs = {}'.format(X.shape))
        return self.down(X)


class Up(nn.Module):
    def __init__(self, size, padding=0):
        super(Up, self).__init__()

        assert(size > 2)
        assert(padding >= 0)

        size2 = 1 << size  # 2^10 = 1024
        size2m1 = 1 << (size - 1)  # 2^9 = 512
        size2p1 = 1 << (size + 1)  # 2^11 = 2048

        padding_top = padding // 2
        padding_bottom = padding - padding_top
        padding_left = padding // 2
        padding_right = padding - padding_left

        self.up = nn.ConvTranspose2d(size2p1, size2, kernel_size=2, stride=2)
        self.pad = nn.ReflectionPad2d(padding=(padding_left, padding_right, padding_top, padding_bottom))
        #self.up = nn.Sequential(
        #    nn.ConvTranspose2d(size2p1, size2, kernel_size=2, stride=2),
        #    nn.ReflectionPad2d(padding=(padding_left, padding_right, padding_top, padding_bottom)))
        self.conv = nn.Sequential(
            nn.Conv2d(size2p1, size2, kernel_size=3),
            nn.BatchNorm2d(size2),
            nn.ReLU(),
            nn.Conv2d(size2, size2, kernel_size=3),
            nn.BatchNorm2d(size2),
            nn.ReLU())

    def forward(self, X, Z):
        X = self.pad(self.up(X))
        X = torch.cat([X, Z], dim=1)

        return self.conv(X)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.input_channels = 4

        self.down1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        # 64  -> 128
        self.down2 = Down(7)
        # 128 -> 256
        self.down3 = Down(8)
        # 256 -> 512
        self.down4 = Down(9)
        # 512 -> 1024
        self.even5 = Down(10)
        # 1024 -> 512
        self.up4 = Up(9, 1)
        # 512  -> 256
        self.up3 = Up(8, 9)
        # 256  -> 128
        self.up2 = Up(7, 8)
        self.up1 = Up(6, 8)
        self.out = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ZeroPad2d(padding=(4, 4, 4, 4))
        )

        downup = nn.Sequential(
            self.down1,
            self.down2,
            self.down3,
            self.down4,
            self.even5,
            self.up4,
            self.up3,
            self.up2,
            self.up1
            )

    def forward(self, X):
        H1 = self.down1(X)
        H2 = self.down2(H1)
        H3 = self.down3(H2)
        H4 = self.down4(H3)
        H5 = self.even5(H4)
        H6 = self.up4(H5, H4)
        H7 = self.up3(H6, H3)
        H8 = self.up2(H7, H2)
        H9 = self.up1(H8, H1)
        out = self.out(H9)
        return out


def init_weights(m):
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)