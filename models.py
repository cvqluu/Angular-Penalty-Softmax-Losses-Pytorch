import torch.nn as nn
import torch.nn.functional as F
from loss_functions import AngularPenaltySMLoss


class ConvBaseline(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvBaseline, self).__init__()
        self.convlayers = ConvNet()
        self.fc_final = nn.Linear(3, num_classes)

    def forward(self, x, embed=False):
        x = self.convlayers(x)
        if embed:
            return x
        x = self.fc_final(x)
        return x

class ConvAngularPen(nn.Module):
    def __init__(self, num_classes=10, loss_type='arcface'):
        super(ConvAM, self).__init__()
        self.convlayers = ConvNet()
        self.adms_loss = AngularPenaltySMLoss(3, num_classes, loss_type=loss_type)

    def forward(self, x, labels=None, embed=False):
        x = self.convlayers(x)
        if embed:
            return x
        L = self.adms_loss(x, labels)
        return L

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=8, stride=1))
        self.fc_projection = nn.Linear(512, 3)

    def forward(self, x, embed=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_projection(x)
        return x
