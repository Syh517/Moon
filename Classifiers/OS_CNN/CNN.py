from torch import nn


class CNN_classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=6),
                                    nn.ReLU(True),
                                    nn.MaxPool1d(kernel_size=6, stride=2))

        self.layer2 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=6),
                                    nn.ReLU(True),
                                    nn.MaxPool1d(kernel_size=6, stride=2))


        self.averagepool = nn.AdaptiveAvgPool1d(1)

        # Расчет входных признаков по формуле Lout=((Lin+2*pading - dilation*(kernel - 1) - 1)/stride) + 1
        self.classifier = nn.Sequential(nn.Linear(in_features=32, out_features=512, bias=True),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(in_features=512, out_features=512, bias=True),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(512, 2))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.averagepool(out)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

