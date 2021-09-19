import os
import glob
import time
from train import train_model
from models.generator import Generator
from models.encoder import Autoencoder
from models.discriminator import Discriminator
from models.classifier import ConvNet
from utils import parameters as p


if __name__ == '__main__':
    if not os.path.exists(p.SAVE_MODEL_DIR):
        os.makedirs(p.SAVE_MODEL_DIR)
    if p.SHOW_TRAIN_ADVERSARIAL_IMAGES is True:
        if not os.path.exists(p.ADVERSARIAL_SAMPLES_DIR):
            os.makedirs(p.ADVERSARIAL_SAMPLES_DIR)
    if p.SHOW_TEST_ADVERSARIAL_IMAGES is True:
        if not os.path.exists(p.TEST_ADVERSARIAL_SAMPLES_DIR):
            os.makedirs(p.TEST_ADVERSARIAL_SAMPLES_DIR)
    if p.SHOW_PERTURBATIONS is True:
        if not os.path.exists(p.PERTURBATIONS_DIR):
            os.makedirs(p.PERTURBATIONS_DIR)

    start_time = time.time()
    ############ Create an instance of G and D class ############
    generator, discriminator, model, AEmodel = Generator(), Discriminator(), ConvNet(), Autoencoder()
    train_model(generator, discriminator, model, AEmodel)






