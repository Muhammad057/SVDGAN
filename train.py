import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from utils.dataloader import train_data_loader
from models.discriminator import compute_gradient_penalty
from utils.visualize import show_adversarial_examples, show_perturbations
from utils import parameters as p


def train_model(generator, discriminator, model, AEmodel):
    """
    Training Process of SVDGAN
    Input: Instances of G, Auto-Encoder, D, Classifier
    Output: Trained Models of G, Auto-Encoder, D, Classifier
            Total Train Time
            Adversarial Images (Required: True)
    """
    D_losses, G_losses, classifier_loss_list, classifier_acc_list, AE_trainloss = [], [], [], [], 0.0

    ########### labels for CNN ############
    onehot_c = torch.zeros(10, 10)
    onehot_c = onehot_c.scatter_(1, torch.LongTensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]).view(10, 1), 1).view(10, 10, 1, 1)

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    generator.cuda(), discriminator.cuda(), model.cuda(), AEmodel.cuda()

    ############ Loss and Optimizer ############
    classifier_criterion = nn.CrossEntropyLoss()
    AE_criterion = nn.MSELoss()

    ########### Adam optimizer ###########
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=p.LR, betas=(p.B1, p.B2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=p.LR, betas=(p.B1, p.B2))
    Classifier_optimizer = optim.Adam(model.parameters(), lr=p.LR, betas=(p.B1, p.B2))
    AE_optimizer = optim.Adam(AEmodel.parameters(), lr=p.LR_AE, betas=(p.B1, p.B2))

    ########### monitor training loss ###########
    batches_done = 0
    print('training start!')
    start_time = time.time()
    for epoch in range(p.N_EPOCHS):

        ########### learning rate decay ###########
        if (epoch+1) == 25:
            optimizer_G.param_groups[0]['lr'] /= 10
            optimizer_D.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        if (epoch+1) == 50:
            optimizer_G.param_groups[0]['lr'] /= 10
            optimizer_D.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        epoch_start_time = time.time()
        for i, (imgs, labels) in enumerate(train_data_loader):

            ################## train Discriminator D & Classifier model ##################
            discriminator.zero_grad()
            model.zero_grad()
            batch_size = imgs.shape[0]

            # Move to GPU if necessary
            real_imgs = imgs.type(Tensor)
            labels = labels.type(LongTensor)
            ################## Calculating D Real Loss D(real_imgs,labels) ##################
            real_validity = discriminator(real_imgs, labels)

            ################## Calculating D Fake Loss D(G(z,labels),y_fill) ##################
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], p.LATENT_DIM)))
            fake_imgs = generator(z, labels)
            fake_imgs.view(-1, p.CHANNELS, p.IMG_SIZE, p.IMG_SIZE)

            ################## Defining Labels for CNN Classifer ##################
            y_label_c = onehot_c[labels] #torch.Size([128, 10, 1, 1])
            y_label_c = torch.mean(y_label_c, -1)
            y_label_c = torch.mean(y_label_c, -1)
            y_label_c = y_label_c.type(torch.int64)
            y_label_c = Variable(y_label_c.cuda())

            ################## Defining AE Model ##################
            real_imgs = real_imgs.view(real_imgs.size(0), -1)
            AE_Result = AEmodel(real_imgs)
            w2 = AEmodel.fc2.weight #fc2 weights 784,16
            w2 = w2.detach().cpu().numpy()
            (p_linear_ae, _, _) = np.linalg.svd(w2, full_matrices=False)
            p_linear_ae = np.reshape(p_linear_ae.T, [16, p.IMG_SIZE, p.IMG_SIZE])    # reshape loading vectors
            perturbation = torch.from_numpy(p_linear_ae[0])
            perturbation = perturbation.unsqueeze(0)
            perturbation = perturbation.unsqueeze(0)
            perturbation = perturbation.cuda()

            perturbed_result = torch.add(fake_imgs, perturbation)
            perturbed_result = perturbed_result.view(perturbed_result.shape[0], -1)

            ################## Defining Losses for D_Fake and CNN Classifer ##################
            fake_validity = discriminator(perturbed_result, labels)
            fake_imgs = fake_imgs.view(fake_imgs.shape[0], 1, p.IMG_SIZE, p.IMG_SIZE)
            real_imgs = real_imgs.view(real_imgs.shape[0], 1, p.IMG_SIZE, p.IMG_SIZE)
            perturbed_result = perturbed_result.view(perturbed_result.shape[0], 1, p.IMG_SIZE, p.IMG_SIZE)

            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs.data, fake_imgs.data, labels.data, cuda)

            classifier_result = model(perturbed_result)
            classifier_loss = classifier_criterion(classifier_result, torch.max(y_label_c, 1)[1])

            ################## Adversarial loss ##################
            D_train_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + p.LAMBDA_GP * gradient_penalty
            D_train_loss.backward(retain_graph=True)
            optimizer_D.step()
            classifier_loss.backward(retain_graph=True)
            Classifier_optimizer.step()
            D_losses.append(D_train_loss.item())

            ################## train generator G ##################
            optimizer_G.zero_grad()
            generator.zero_grad()
            y_label_c = onehot_c[labels] #torch.Size([128, 10, 1, 1])
            y_label_c = torch.mean(y_label_c, -1)
            y_label_c = torch.mean(y_label_c, -1)
            y_label_c = y_label_c.type(torch.int64)
            y_label_c = Variable(y_label_c.cuda())

            ################## Train the generator every n_critic steps ##################
            if i % p.N_CRITIC == 0:

                # Generate a batch of fake images
                fake_imgs = generator(z, labels)
                fake_imgs.view(-1, p.CHANNELS, p.IMG_SIZE, p.IMG_SIZE)

                AEmodel.zero_grad()
                real_imgs = real_imgs.view(real_imgs.size(0), -1)
                AE_Result = AEmodel(real_imgs)
                w2 = AEmodel.fc2.weight #fc2 weights 784,16
                w2 = w2.detach().cpu().numpy()
                (p_linear_ae, _, _) = np.linalg.svd(w2, full_matrices=False)
                p_linear_ae = np.reshape(p_linear_ae.T, [16, p.IMG_SIZE, p.IMG_SIZE])    # reshape loading vectors
                perturbation = torch.from_numpy(p_linear_ae[0])
                perturbation = perturbation.unsqueeze(0)
                perturbation = perturbation.unsqueeze(0)
                perturbation = perturbation.cuda()
                perturbed_result = torch.add(fake_imgs, perturbation)

                fake_validity = discriminator(perturbed_result, labels)
                G_loss = -torch.mean(fake_validity)

                classifier_result = model(perturbed_result)
                classifier_loss = classifier_criterion(classifier_result, torch.max(y_label_c, 1)[1])
                classifier_loss_list.append(classifier_loss.item())

                ################## Track the CNN accuracy ##################
                total = y_label_c.size(0)
                _, predicted = torch.max(classifier_result.data, 1)
                correct = (predicted == torch.max(y_label_c, 1)[1]).sum().item()
                classifier_acc_list.append((correct / total) * 100)

                G_train_loss = G_loss + 0.5 * classifier_loss # 0.5 is hyper-parameter
                G_train_loss.backward()
                optimizer_G.step()

                G_losses.append(G_train_loss.item())
                AE_loss = AE_criterion(AE_Result, real_imgs)
                AE_loss.backward(retain_graph=True)
                AE_optimizer.step()
                AE_trainloss += AE_loss.item()*real_imgs.size(0)

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, p.N_EPOCHS, i, len(train_data_loader), D_train_loss.item(), G_loss.item()))

                print('[Epoch %d/%d] [Batch %d/%d], [loss_CNN: %.4f], [accuracy_CNN: %.2f]'
                      % (epoch, p.N_EPOCHS, i, len(train_data_loader), torch.mean(torch.FloatTensor(classifier_loss_list)),
                         torch.mean(torch.FloatTensor(classifier_acc_list))))

                batches_done += p.N_CRITIC

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

        train_loss = AE_trainloss/len(train_data_loader)
        print("AELoss", train_loss)
        if p.SHOW_TRAIN_ADVERSARIAL_IMAGES is True:
            show_adversarial_examples(epoch, fake_imgs, AE_Result, perturbed_result)
        if p.SHOW_PERTURBATIONS is True:
            show_perturbations(epoch, p_linear_ae)
        end_time = time.time()
    total_ptime = end_time - start_time
    print("Training finished in %f minutes" % (total_ptime/60))
    print("save training models")

    torch.save(generator.state_dict(), p.SAVE_MODEL_DIR + 'generator_param.pkl')
    torch.save(AEmodel.state_dict(), p.SAVE_MODEL_DIR + 'encoder_param.pkl')
    torch.save(discriminator.state_dict(), p.SAVE_MODEL_DIR + 'discriminator_param.pkl')
    torch.save(model.state_dict(), p.SAVE_MODEL_DIR + 'classifier_param.pkl')

