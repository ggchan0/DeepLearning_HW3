import torch
import torch.nn.functional as F
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        kernel = 5
        layers = [16, 32, 64, 128]
        L = []
        c = 6
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel, stride=2, padding=kernel//2))
            L.append(torch.nn.BatchNorm2d(num_features = l))
            L.append(torch.nn.ReLU())
            c = l

        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 6)

    def forward(self, x):
        return self.classifier(self.network(x))

class FCN(torch.nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        self.conv0 = nn.Conv2d(3, 36, 5, padding = 2)
        self.conv1 = nn.Conv2d(36, 24, 5, padding = 2)
        self.conv2 = nn.Conv2d(24, 12, 5, padding = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(12, 24, 5, padding = 2)
        self.conv4 = nn.Conv2d(24, 36, 5, padding = 2)
        self.linear1 = nn.Linear(36 * 48 * 64, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 6)

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
        out_c0 = F.relu(self.conv0(x))
        out_c1 = F.relu(self.conv1(out_c0))
        out_c2 = F.relu(self.conv2(out_c1))
        out_c3 = F.relu(self.conv3(out_c2)) + out_c1
        out_c4 = F.relu(self.conv4(out_c3)) + out_c0
        out_pool1 = F.relu(self.pool(out_c4))
        out_pool1 = out_pool1.view(-1, 36 * 48 * 64)
        out_l1 = self.linear1(F.relu(out_pool1))
        out_l2 = self.linear2(F.relu(out_l1))
        out_l3 = self.linear3(F.relu(out_l2))
        return out_l3


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
