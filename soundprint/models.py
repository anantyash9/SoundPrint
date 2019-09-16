from typing import List, Callable
import torch
from torch import nn
from torch.nn import functional as F

class AdditiveSoftmaxLinear(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, scale: float = 30, margin: float = 0.35):
        super(AdditiveSoftmaxLinear, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.Tensor(input_dim, num_classes), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight)


    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None):
        if self.training:
            assert labels is not None
            # Normalise embeddings
            embeddings = embeddings.div((embeddings.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8))

            # Normalise weights
            normed_weights = self.weight / (self.weight.pow(2).sum(dim=0, keepdim=True).sqrt() + 1e-8)

            cos_theta = torch.mm(embeddings, normed_weights)
            phi = cos_theta - self.margin

            labels_onehot = F.one_hot(labels, self.num_classes).byte()
            logits = self.scale * torch.where(labels_onehot, phi, cos_theta)

            return logits
        else:
            assert labels is None
            return torch.mm(embeddings, self.weight)

class GlobalAvgPool1d(nn.Module):
    def forward(self, input):
        return F.avg_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: int = None,
                 conv_type: Callable = nn.Conv1d):
        super(ResidualBlock1D, self).__init__()
        self.conv_type = conv_type
        self.conv1 = self.conv_type(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = self.conv_type(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        if stride > 1:
            self.downsample = True
            self.downsample_op = nn.Sequential(
                self.conv_type(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
                nn.ReLU()
            )
        else:
            self.downsample = False

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_op(x)

        out += identity
        out = F.relu(out)

        return out



class ResidualClassifier(nn.Module):
    def __init__(self, in_channels: int, filters: int, layers: List[int], num_classes: int, dim: int = 1):
        super(ResidualClassifier, self).__init__()
        self.filters = filters
        self.dim = dim
        self.conv_type = nn.Conv1d if dim == 1 else nn.Conv2d

        self.conv1 = self.conv_type(in_channels, filters, kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.filters)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.filters, filters, layers[0])
        self.layer2 = self._make_layer(filters, filters*2, layers[1], stride=2)
        self.layer3 = self._make_layer(filters*2, filters*4, layers[2], stride=2)
        self.layer4 = self._make_layer(filters*4, filters*8, layers[3], stride=2)
        self.avgpool = GlobalAvgPool1d()
        self.layer_norm = nn.LayerNorm(filters*8, elementwise_affine=False)

        self.fc = AdditiveSoftmaxLinear(filters*8, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride, conv_type=self.conv_type))

        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, conv_type=self.conv_type))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, return_embedding: bool = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Normalise embeddings
        x = self.layer_norm(x)

        if return_embedding:
            return x
        else:
            x = self.fc(x, y)
            return x
