import torch.nn as nn


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
    def __init__(self, in_channel, filters, patch_size):
        super(ConvStem, self).__init__()
        self.conv = nn.Conv2d(in_channel, filters, kernel_size=patch_size, stride=patch_size)
        self.activation = ActivationBlock(filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvMixerBlock(nn.Module):
    def __init__(self, column, level, filters, kernel_size):
        super(ConvMixerBlock, self).__init__()
        self.column = column
        self.level = level
        self.depthwise_conv = nn.Conv2d(
            filters,
            filters,
            kernel_size=kernel_size,
            padding='same',
            groups=filters,
        )
        self.pointwise_conv = nn.Conv2d(filters, filters, kernel_size=1)
        self.activation_block = ActivationBlock(filters)
        self.activation_block2 = ActivationBlock(filters)

    def reverasable_output_array(self, x, r_arr, column, level):
        r_arr[column][level] = x
        return r_arr
    
    def forward(self, data):
        x, pe_block, r_arr = data
        if self.column == 0:
            x0 = x
            x = self.depthwise_conv(x)
            x = self.activation_block(x) + x0
            x = self.pointwise_conv(x)
            x = self.activation_block2(x)
            r_arr = self.reverasable_output_array(x, r_arr, self.column, self.level)
            return x, pe_block, r_arr
        else:
            if self.level == 0:
                x0 = r_arr[self.column - 1][self.level + 1] + pe_block
            elif self.level < len(r_arr[0]) - 1 and self.level != 0:
                x0 = r_arr[self.column - 1][self.level + 1] + x
            else:
                x0 = x
            x = self.depthwise_conv(x)
            x = self.activation_block(x) + x0
            x = self.pointwise_conv(x)
            x = self.activation_block2(x)

            if self.column != 0:
                x = r_arr[self.column - 1][self.level] + x
            
            r_arr = self.reverasable_output_array(x, r_arr, self.column, self.level)
            return x, pe_block, r_arr


class Siren(nn.Module):
    def __init__(
        self, in_channels, filters, depth, columns, kernel_size, patch_size, n_classes
    ):
        super(Siren, self).__init__()
        self.depth = depth
        self.columns = columns
        self.stem = ConvStem(in_channels, filters, patch_size)

        self.mixer_blocks = nn.Sequential()
        for column in range(columns):
            for level in range(depth):
                self.mixer_blocks.add_module(
                    f"block_{column}{level}", ConvMixerBlock(column, level, filters, kernel_size)
                )

        self.classification_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            # nn.Linear(filters, n_classes),
        )

    def forward(self, x):
        r_arr = [[None for _ in range(self.depth)] for _ in range(self.columns)]
        x = self.stem(x)
        pe_block = x
        x, _, _ = self.mixer_blocks((x, pe_block, r_arr))
        x = self.classification_block(x)
        return x    


def get_model(in_channels, filters, depth, columns, kernel_size, patch_size, n_classes):
    model = Siren(
        in_channels=in_channels,
        filters=filters,
        depth=depth,
        columns=columns,
        kernel_size=kernel_size,
        patch_size=patch_size,
        n_classes=n_classes
    )
    return model

        