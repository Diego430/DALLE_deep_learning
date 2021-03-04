from torch import nn


# Istanzia un blocco di ResNet con 5 layer, 3 3D convoluzionali di dimensione [chan, chan, 3], con chan = numero canali, e padding 1 e 2 ReLu
class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x
