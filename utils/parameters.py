########### training parameters ###########

BATCH_SIZE = 64
LR = 0.0002
B1 = 0.5
B2 = 0.999
LR_AE = 0.0001
N_EPOCHS = 25
IMG_SIZE = 28
CHANNELS = 1
ENCODING_DIM = 16
N_CRITIC = 5
LAMBDA_GP = 10
CLIP_VALUE = 0.01
SAMPLE_INTERVAL = 100
N_CLASSES = 10
LATENT_DIM = 100
IMAGE_SHAPE = (CHANNELS, IMG_SIZE, IMG_SIZE)

SHOW_TRAIN_ADVERSARIAL_IMAGES = False
SHOW_TEST_ADVERSARIAL_IMAGES = False
SHOW_PERTURBATIONS = False


TRAIN_DIR = r'/home/aai/Documents/My_Research_Articles/Generating Low-Cost Adv. Attacks/dataset/MNIST/training'
TEST_DIR = r'/home/aai/Documents/My_Research_Articles/Generating Low-Cost Adv. Attacks/dataset/MNIST/testing'
SAVE_MODEL_DIR = r'/home/aai/Documents/My_Research_Articles/SVDGAN/output/trained_models/'
ADVERSARIAL_SAMPLES_DIR = r'/home/aai/Documents/My_Research_Articles/SVDGAN/output/train_adversarial_samples/'
TEST_ADVERSARIAL_SAMPLES_DIR = r'/home/aai/Documents/My_Research_Articles/SVDGAN/output/test_adversarial_samples/'
PERTURBATIONS_DIR = r'/home/aai/Documents/My_Research_Articles/SVDGAN/output/perturbations/'
