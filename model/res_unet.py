import torch
import torch.nn as nn


__all__ = ['UResNet', 'init_weights']


class UResConv(nn.Module):
    def __init__(self, size, from_size=None):
        super(UResConv, self).__init__()

        assert(size > 2)

        if from_size is None:
            from_size = size

        from_size2 = 1 << from_size
        size2 = 1 << size
        size2m1 = 1 << (size - 1)

        self.conv = nn.Sequential(
            nn.Conv2d(from_size2, size2, kernel_size=3, padding=1),
            nn.BatchNorm2d(size2, track_running_stats=False),
            nn.ReLU(),
        )

    def forward(self, X):
        return self.conv(X)


class UResBlock(nn.Module):
    def __init__(self, size, from_size=None):
        super(UResBlock, self).__init__()

        assert(size > 2)

        if from_size is None:
            from_size = size

        self.conv = nn.Sequential(
            UResConv(size, from_size),
            UResConv(size),
        )

    def forward(self, X):
        return self.conv(X)


class UResDown(nn.Module):
    def __init__(self, size):
        super(UResDown, self).__init__()
        assert(size > 2)
        size2 = 1 << size
        size2m1 = 1 << (size - 1)
        size2p1 = 1 << (size + 1)

        use_max_pool = False

        if use_max_pool:
            self.down = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(size2m1, size2, kernel_size=2),
            )
        else:
            self.down = nn.Conv2d(size2m1, size2, kernel_size=2, stride=2)

        self.conv = UResBlock(size)

    def forward(self, X):
        Xd = self.down(X)
        Xp = self.conv(Xd)
        return Xp + Xd


class UResUp(nn.Module):
    def __init__(self, size):
        super(UResUp, self).__init__()

        assert(size > 2)

        size2 = 1 << size
        size2p1 = 1 << (size + 1)

        upsample = False
        if upsample is True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(size2p1, size2, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(size2p1, size2, kernel_size=2, stride=2)

        self.shortcut = nn.Conv2d(size2p1, size2, kernel_size=1)

        self.conv = UResBlock(size, size + 1)

    def forward(self, X, Z):
        Xu = self.up(X)
        Xcat = torch.cat([Xu, Z], dim=1)
        Xp = self.conv(Xcat)

        return Xp + self.shortcut(Xcat)


class UResNet(nn.Module):
    def __init__(self, input_channels=4):
        super(UResNet, self).__init__()

        self.input_channels = input_channels

        base_size = 6
        base_size2 = 1 << base_size2
        self.pre = nn.Sequential(
            nn.Conv2d(self.input_channels, base_size2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_size2, track_running_stats=False),
            nn.ReLU(),
        )
        self.down1 = UResBlock(base_size)
        # 64  -> 128
        self.down2 = UResDown(7)
        # 128 -> 256
        self.down3 = UResDown(8)
        # 256 -> 512
        self.down4 = UResDown(9)
        # 512 -> 1024
        self.even5 = UResDown(10)
        # 1024 -> 512
        self.up4 = UResUp(9)
        # 512  -> 256
        self.up3 = UResUp(8)
        # 256  -> 128
        self.up2 = UResUp(7)
        self.up1 = UResUp(6)
        self.out = nn.Sequential(
            nn.Conv2d(base_size, 1, kernel_size=1)
        )

    def forward(self, X):
        Xp = self.pre(X)
        H1 = Xp + self.down1(Xp)
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

