import time
import torch
import numpy as np
from models.generator import Generator
from models.encoder import Autoencoder
from models.classifier import ConvNet
from utils.dataloader import test_data_loader
from utils.visualize import show_test_adversarial_examples
from utils import parameters as p


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

########### Load Pre-trained Models ###########
generator, model, AEmodel = Generator(), ConvNet(), Autoencoder()
generator.load_state_dict(torch.load(p.SAVE_MODEL_DIR + "generator_param.pkl"))
model.load_state_dict(torch.load(p.SAVE_MODEL_DIR + "classifier_param.pkl"))
AEmodel.load_state_dict(torch.load(p.SAVE_MODEL_DIR + "encoder_param.pkl"))
generator.eval(), AEmodel.eval(), model.eval()
generator.cuda(), AEmodel.cuda(), model.cuda()

onehot_c = torch.zeros(10, 10)
onehot_c = onehot_c.scatter_(1, torch.LongTensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]).view(10, 1), 1).view(10, 10, 1, 1)

correct = 0
total = 0
with torch.no_grad():
    for i, (imgs, labels) in enumerate(test_data_loader):
        start_time = time.time()
        real_imgs = imgs.type(Tensor)
        labels = labels.type(LongTensor)
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], p.LATENT_DIM)))
        fake_img = generator(z, labels)
        real_imgs = real_imgs.view(real_imgs.size(0), -1)
        AE_Result = AEmodel(real_imgs)
        w2 = AEmodel.fc2.weight
        w2 = w2.detach().cpu().numpy()
        (p_linear_ae, _, _) = np.linalg.svd(w2, full_matrices=False)
        p_linear_ae = np.reshape(p_linear_ae.T, [16, p.IMG_SIZE, p.IMG_SIZE]) # reshape loading vectors before plotting
        perturbation = torch.from_numpy(p_linear_ae[0])
        perturbation = perturbation.unsqueeze(0)
        perturbation = perturbation.unsqueeze(0)
        perturbation = perturbation.cuda()

        fake_img = fake_img.view(-1, p.CHANNELS, p.IMG_SIZE, p.IMG_SIZE)
        perturbed_result = torch.add(fake_img, perturbation)
        if p.SHOW_TEST_ADVERSARIAL_IMAGES is True:
            show_test_adversarial_examples(i, fake_img, perturbed_result)

        y_label_ = onehot_c[labels] #torch.Size([1, 10, 1, 1])
        y_label_ = torch.mean(y_label_, -1)
        y_label_ = torch.mean(y_label_, -1) #torch.Size([1, 10])
        y_label_ = y_label_.type(torch.int64)
        y_label_ = y_label_.cuda()
        outputs = model(perturbed_result) #torch.Size([1, 10])
        _, predicted = torch.max(outputs.data, 1)

        total += y_label_.size(0)
        correct += (predicted == torch.max(y_label_, 1)[1]).sum().item() # targeted
        print("Time to produce a single attack instance is: %s seconds" % ((time.time()-start_time)*1000))
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

