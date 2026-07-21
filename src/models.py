from torch import Tensor, nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,),
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(
            in_features=128,
            out_features=num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, 3, 224, 224]
        x = self.features(x)
        # x: [batch_size, 128, 28, 28]

        x = self.pool(x)
        # x: [batch_size, 128, 1, 1]

        x = x.flatten(start_dim=1)
        # x: [batch_size, 128]

        logits = self.classifier(x)
        # logits: [batch_size, num_classes]

        return logits
    
