import matplotlib.pyplot as plt
from utils import parameters as p


def show_adversarial_examples(epoch, fake_result, encoder_result, perturbed_result):
    """
    :param epoch:
    :param fake_result:
    :param encoder_result:
    :param perturbed_result:
    :return: images stored in a folder:
              1. fakes images from generator -> fake_result
              2. auto-encoder images -> encoder_result
              3. adversarial images -> perturbed_result
    """
    filename_fake = p.ADVERSARIAL_SAMPLES_DIR + "fake_image_%d.png" % epoch
    plt.imsave(filename_fake, fake_result[epoch].cpu().data.view(28, 28).numpy(), cmap='gray')

    filename_encoder = p.ADVERSARIAL_SAMPLES_DIR + "AE_image_%d.png" % epoch
    plt.imsave(filename_encoder, encoder_result[epoch].cpu().data.view(28, 28).numpy(), cmap='gray')

    filename_perturbed = p.ADVERSARIAL_SAMPLES_DIR + "perturbed_image_%d.png" % epoch
    plt.imsave(filename_perturbed, perturbed_result[epoch].cpu().data.view(28, 28).numpy(), cmap='gray')


def show_test_adversarial_examples(epoch, fake_result, perturbed_result):
    """
    :param epoch:
    :param fake_result:
    :param perturbed_result:
    :return: fake images and adversarial images stored in a folder
    """
    filename_fake = p.TEST_ADVERSARIAL_SAMPLES_DIR + "original_image%d.png" % epoch
    plt.imsave(filename_fake, fake_result.cpu().data.view(28, 28).numpy(), cmap='gray')

    filename_perturbed = p.TEST_ADVERSARIAL_SAMPLES_DIR + "perturbed_image%d.png" % epoch
    plt.imsave(filename_perturbed, perturbed_result.cpu().data.view(28, 28).numpy(), cmap='gray')


def show_perturbations(epoch, perturbations):
    """
    :param epoch:
    :param perturbations:
    :return: perturbations stored in a folder
    """
    count = 0
    while count < 16:
        filename_perturbations = p.PERTURBATIONS_DIR + "perturbation_%d_%d.png" % (epoch, count)
        plt.imsave(filename_perturbations, perturbations[count], cmap='gray')
        count += 1
