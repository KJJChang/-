from torch import nn


class CNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=8,
                kernel_size=(3,3),
                stride=1,
                padding=1
                ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3,3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3,3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3,3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=16*7*7, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x
