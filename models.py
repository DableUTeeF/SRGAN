from torch import nn
from torch.nn import functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return out


class UpsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels, _):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels*4, 3, stride=1, padding=1, bias=False)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffler(x)
        return F.relu(x)


def make_block(block, in_planes, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
        layers.append(block(in_planes, planes, stride))
        in_planes = planes
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, _):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.bn2(x)
        return F.leaky_relu(x, 0.2)


class Generator(nn.Module):
    """
    The generator network for generate SR Images
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, bias=False)
        self.res_blocks = make_block(BasicBlock, 64, 64, 5, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # self.upsampling_blocks = make_block(UpsampleBlock, 64, 256, 2, None)
        self.upsampling_blocks = UpsampleBlock(64, 256, None)
        self.conv3 = nn.Conv2d(256, 3, kernel_size=9, stride=1, padding=4, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        out = F.relu(x)
        out = self.res_blocks(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.upsampling_blocks(out)
        return self.conv3(out)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(64, 128, None)
        self.conv_block2 = ConvBlock(128, 256, None)
        self.conv_block3 = ConvBlock(256, 512, None)
        self.fc1 = nn.Linear(512, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        return F.sigmoid(x)
