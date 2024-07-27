import torch.nn as nn
from .mixstyle import MixStyle

class ActivationBlock(nn.Module):
    def __init__(self, num_features):
        super(ActivationBlock, self).__init__()
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        x = self.activation(x)
        x = self.batch_norm(x)
        return x

class ConvStem(nn.Module):
    def __init__(self, in_channel, filter, patch_size):
        super(ConvStem, self).__init__()
        self.conv = nn.Conv2d(in_channel, filter, kernel_size=patch_size, stride=patch_size)
        self.activation_block = ActivationBlock(filter)


    def forward(self, x):
        x = self.conv(x)
        x = self.activation_block(x)
        return x

class ConvMixerBlock(nn.Module):
    def __init__(self, filter, kernel_size):
        super(ConvMixerBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            filter,
            filter,
            kernel_size=kernel_size,
            padding = 'same',
            groups = filter
        )
        self.pointwise_conv = nn.Conv2d(filter, filter, kernel_size=1)
        self.activation_block = ActivationBlock(filter)
        self.activation_block2 = ActivationBlock(filter)

    def forward(self, x):
        x0 = x
        x = self.depthwise_conv(x)
        x = self.activation_block(x) + x0
        x = self.pointwise_conv(x)
        x = self.activation_block2(x)
        return x

class ConvMixer(nn.Module):
    def __init__(self, in_channels, filter, depth, kernel_size, patch_size, n_classes):
        super(ConvMixer, self).__init__()
        self.in_channels = in_channels
        self.filter = filter
        self.depth = depth
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.mix_layers = [0,1]

        self.stem = ConvStem(in_channels, filter, patch_size)

        self.mixer_blocks = nn.Sequential()
        for i in range(depth):
            self.mixer_blocks.add_module(
                f"block_{i}", ConvMixerBlock(filter, kernel_size)
            )

            # add mix style blocks
            if i in self.mix_layers:
                self.mixer_blocks.add_module(
                    f"mix_{i}", MixStyle(p=0.4, alpha=0.1, mix='random')
                )

        self.classification_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(filter, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.mixer_blocks(x)
        x = self.classification_block(x)
        return x

def get_model_mix(in_channels, filter, depth, kernel_size, patch_size, n_classes):
    model = ConvMixer(
        in_channels=in_channels,
        filter=filter,
        depth=depth,
        kernel_size=kernel_size,
        patch_size=patch_size,
        n_classes=n_classes
    )
    return model