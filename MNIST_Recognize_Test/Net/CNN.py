import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            # 输入[batch_size, 1, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # 输出[batch_size, 32, 28, 28]
            # BN标准化，不会改变输出的尺寸
            nn.BatchNorm2d(32, affine=True, eps=1e-05, track_running_stats=True),
            # 激活函数
            nn.ReLU(inplace=False),
            # 输入[batch_size, 32, 28, 28]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # 输出[batch_size, 64, 28, 28]
            nn.BatchNorm2d(64, affine=True, eps=1e-05, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            # 输出[batch_size, 64, 14, 14]
        )
        self.fcn = nn.Sequential(
            nn.Linear(in_features=64*14*14, out_features=2048, bias=True),
            nn.Linear(in_features=2048, out_features=10, bias=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        y = self.fcn(x)
        return y
