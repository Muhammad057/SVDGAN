import torch
import torch.nn as nn
import numpy as np
from utils import parameters as p


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(p.N_CLASSES, p.N_CLASSES)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(p.LATENT_DIM + p.N_CLASSES, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(p.IMAGE_SHAPE))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        img = self.model(gen_input)
        img = img.view(img.shape[0], p.CHANNELS, p.IMG_SIZE, p.IMG_SIZE)
        return img
