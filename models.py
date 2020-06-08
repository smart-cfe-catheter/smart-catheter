import torch
import torchvision
from torch import nn
from torch.nn import functional as f


def stack_linear_layers(dim, layer_cnt):
    return [nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.LeakyReLU()) for _ in range(layer_cnt)]


class ResNet(nn.Module):
    def __init__(self, output_stride=16):
        super(ResNet, self).__init__()
        self.output_stride = output_stride
        if output_stride == 8:
            self.grid = [1, 2, 1]
            self.stride = [1, 2, 1, 1]
            self.dilation = [1, 1, 2, 4]
        elif output_stride == 16:
            self.grid = [1, 2, 4]
            self.stride = [1, 2, 2, 1]
            self.dilation = [1, 1, 1, 2]

    def parse_model(self):
        self.stem = nn.Sequential(*list(self.base_model.children())[:4])
        self.block1 = self.base_model.layer1
        self.block2 = self.base_model.layer2
        self.block3 = self.base_model.layer3
        self.block4 = self.base_model.layer4
        blocks = [self.block1, self.block2, self.block3, self.block4]

        for i, block in enumerate(blocks):
            if i != 3:
                for bottleneck in block:
                    if bottleneck.downsample is not None:
                        bottleneck.downsample[0].stride = (self.stride[i], self.stride[i])
                        bottleneck.conv2.stride = (self.stride[i], self.stride[i])
                    bottleneck.conv2.dilation = (self.dilation[i], self.dilation[i])
                    bottleneck.conv2.padding = self.get_padding_size(bottleneck.conv2.kernel_size,
                                                                     bottleneck.conv2.dilation)

            else:
                for bottleneck, grid in zip(block, self.grid):
                    if bottleneck.downsample is not None:
                        bottleneck.downsample[0].stride = (self.stride[i], self.stride[i])
                        bottleneck.conv2.stride = (self.stride[i], self.stride[i])
                    bottleneck.conv2.dilation = (grid * self.dilation[i], grid * self.dilation[i])
                    bottleneck.conv2.padding = self.get_padding_size(bottleneck.conv2.kernel_size,
                                                                     bottleneck.conv2.dilation)
        del self.base_model

    @staticmethod
    def get_padding_size(kernel_size, dilation):
        width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
        height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        return pad_size

    def forward(self, x):
        x = self.stem(x)
        low_level_feature = self.block1(x)
        out = self.block2(low_level_feature)
        out = self.block3(out)
        out = self.block4(out)

        return low_level_feature, out


class ResNet152(ResNet):
    def __init__(self, pretrained, output_stride=16):
        super().__init__(output_stride=output_stride)
        self.base_model = torchvision.models.resnet152(pretrained=pretrained)
        self.parse_model()


class DNN(nn.Module):
    def __init__(self, layer_cnt):
        super(DNN, self).__init__()
        self.type = 'DNN'

        layers = stack_linear_layers(3, layer_cnt)
        self.fc_layers = nn.Sequential(*layers)
        self.decoder = nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc_layers(x)
        x = f.leaky_relu(self.decoder(x))
        return x


class RNN(nn.Module):
    def __init__(self, layer_cnt):
        super(RNN, self).__init__()
        self.type = 'RNN'

        self.num_layers = layer_cnt
        self.rnn = nn.GRU(3, 100, self.num_layers, batch_first=True)
        self.decoder = nn.Linear(3, 1)

    def forward(self, x, h=None):
        x, h = self.rnn(x, h)
        x = self.decoder(x)

        return x, h

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, 3)


class CNN(nn.Module):
    def __init__(self, layer_cnt):
        super(CNN, self).__init__()
        self.type = 'CNN'

        self.backbone = ResNet152(pretrained=False)

    def forward(self, x):
        return self.backbone(x)
