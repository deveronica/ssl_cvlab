import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, cfg):
        super(ResNet18, self).__init__()
        self.cfg = cfg
        self.model = models.resnet18(num_classes=self.cfg.num_classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, cfg.num_classes)
        if cfg.in_channels != 3:  # 입력 채널이 3이 아닌 경우
            self.model.conv1 = nn.Conv2d(cfg.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self