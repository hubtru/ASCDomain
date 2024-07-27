import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(in_channels, filter, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(in_channels, filter, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(filter),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(filter, filter, kernel_size, groups=filter, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(filter)
                )),
                nn.Conv2d(filter, filter, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(filter)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(filter, n_classes)
    )

def get_model(in_channels, filter, depth, kernel_size, patch_size, n_classes):
    model = ConvMixer(
        in_channels=in_channels,
        filter=filter,
        depth=depth,
        kernel_size=kernel_size,
        patch_size=patch_size,
        n_classes=n_classes
    )
    return model