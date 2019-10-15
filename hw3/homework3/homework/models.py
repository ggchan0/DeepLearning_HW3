import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from . import dense_transforms


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CNNLayer,self).__init__()
        L = []
        L.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        L.append(nn.BatchNorm2d(out_channels))
        L.append(nn.ReLU())
        L.append(nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        L.append(nn.BatchNorm2d(out_channels))
        L.append(nn.ReLU())
        L.append(nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        L.append(nn.BatchNorm2d(out_channels))
        L.append(nn.ReLU())
        L.append(nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        L.append(nn.BatchNorm2d(out_channels))
        L.append(nn.ReLU())
        L.append(nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.network = nn.Sequential(*L)

    def forward(self, x):
        return self.network(x)


class CNNClassifier(nn.Module):
    def __init__(self, layers = [16, 32, 64, 128]):
        super(CNNClassifier, self).__init__()
        #layers = [32, 64, 128, 256]
        layers = [256, 128, 64, 32]
        L = []

        L.append(nn.Conv2d(3, layers[0], kernel_size = 7, stride = 2, padding = 7 // 2))
        L.append(nn.BatchNorm2d(layers[0]))
        L.append(nn.ReLU())
        #L.append(nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))

        c = layers[0]
        for l in layers:
            L.append(CNNLayer(c, l, 2))
            c = l

        self.network = nn.Sequential(*L)
        self.classifier = nn.Linear(c, 6)

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=[2,3])
        return self.classifier(x)

class FCN(torch.nn.Module):
    class ConvLayer(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            L = []
            L.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
            L.append(nn.BatchNorm2d(out_channels))
            L.append(nn.ReLU())
            L.append(nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
            L.append(nn.BatchNorm2d(out_channels))
            L.append(nn.ReLU())
            L.append(nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
            self.network = nn.Sequential(*L)

        def forward(self, x):
            return self.network(x)

    class TransposeLayer(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            L = []
            L.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding=1, dilation=1))
            L.append(nn.BatchNorm2d(out_channels))
            L.append(nn.ReLU())
            self.network = nn.Sequential(*L)

        def forward(self, x):
            return self.network(x)

    def __init__(self, layers=[16, 32, 64, 128], kernel_size=3):
        super(FCN, self).__init__()
        transforms = dense_transforms.Compose([
            dense_transforms.Resize((96, 128)),
            dense_transforms.RandomHorizontalFlip(),
            dense_transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            dense_transforms.ToTensor(),
            dense_transforms.Normalize([0.425, 0.425, 0.425], [0.25, 0.25, 0.25])
        ])
        layers = [32, 64, 128, 256]


        L = []

        L.append(nn.Conv2d(3, layers[0], kernel_size = 7, stride = 2, padding = 7 // 2))
        L.append(nn.BatchNorm2d(layers[0]))
        L.append(nn.ReLU())

        self.start = nn.Sequential(*L)

        self.conv64 = self.ConvLayer(32, 64)
        self.conv128 = self.ConvLayer(64, 128)
        self.conv256 = self.ConvLayer(128, 256)

        self.trans256 = self.TransposeLayer(256, 128)
        self.trans128 = self.TransposeLayer(128, 64)
        self.trans64 = self.TransposeLayer(64, 32)

        self.trans32 = self.TransposeLayer(32, 5)

        self.norm = dense_transforms.Normalize([0.425, 0.425, 0.425], [0.25, 0.25, 0.25])

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """


        out_32 = None
        out_64 = None
        out_128 = None
        out_256 = None

        # 3 x 128 x 96
        x = self.start(x)
        # 32 x 64 x 48
        out_32 = x

        x = self.conv64(x)
        # 64 x 32 x 24
        out_64 = x

        x = self.conv128(x)
        # 128 x 16 x 12
        out_128 = x

        x = self.conv256(x)
        # 256 x 8 x 6
        out_256 = x

        x = self.trans256(x) + out_128 #128 x 16 x 12
        x = self.trans128(x) + out_64 #64 x 32 x 24
        x = self.trans64(x) + out_32 #32 x 64 x 48

        x = self.trans32(x) #6 x 128 x 96

        return x


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
