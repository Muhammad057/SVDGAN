import torch.nn as nn
from utils import parameters as p

############ Create Linear Auto-Encoder ############
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        ## encoder ##
        self.fc1 = nn.Linear(p.IMG_SIZE * p.IMG_SIZE, p.ENCODING_DIM)

        ## decoder ##
        self.fc2 = nn.Linear(p.ENCODING_DIM, p.IMG_SIZE * p.IMG_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x