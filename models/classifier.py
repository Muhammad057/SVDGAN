import torch.nn as nn

############ Create the CNN Model ############
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats = True))

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    ############ Perform Forward Pass ############
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out