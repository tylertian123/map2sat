import load_dataset
import model

from torch import nn, manual_seed, ones_like, zeros_like, optim, save, cuda
from torch.utils.data import DataLoader
from numpy import random, zeros, savetxt

class GeneratorLoss(nn.Module):
    def __init__(self, baseline_model: bool=True):
        super(GeneratorLoss, self).__init__()
        manual_seed(0)
        self.baseline = baseline_model
        self.coeff = 1000
        self.sig_ce = nn.BCEWithLogitsLoss()
        self.mae = nn.L1Loss()
    
    def forward(self, disc_generator_result, generated_image, label):
        """
        The loss function for the generator

        Arguments:
            disc_generator_result: the result of discriminator for comparing the generated image with input
            generated_image: the generated image from the Generator [batch_size, 3, 256, 256]
            label: the true label of the input (the target)

        Return:
            total_gen_loss: the loss of the generator
        """
        if self.baseline == False:
            real_grid = ones_like(disc_generator_result)
            gan_loss = self.sig_ce(real_grid, disc_generator_result)   # the generator should be good enought that the gan classifies it as real
        l1_loss = self.mae(label, generated_image)   # reconstruction error

        total_gen_loss = (self.coeff * l1_loss) if self.baseline else gan_loss + (self.coeff * l1_loss)

        return total_gen_loss

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        manual_seed(0)
        self.sig_ce = nn.BCEWithLogitsLoss()
    
    def forward(self, disc_label_result, disc_generator_result):
        """
        The loss function for the discriminator

        Arguments:
            disc_label_result: the result of discriminator for comparing the label with input
            disc_generator_result: the result of discriminator for comparing the generated image with input

        Return:
            total_disc_loss: the loss of the discriminator
        """
        fake_grid = zeros_like(disc_generator_result)
        real_grid = ones_like(disc_label_result)

        generated_loss = self.sig_ce(fake_grid, disc_generator_result)   # the discriminator has to classify the generated image as fake
        real_loss = self.sig_ce(real_grid, disc_label_result)   # the discriminator has to classify the label image as real

        total_disc_loss = generated_loss + real_loss

        return total_disc_loss
    
def evaluate(networks: tuple, valid_data: DataLoader, criterions: tuple, baseline_model: bool=True, use_cuda: bool=True):
    """
    Evaluate the model(s) based on the validation data

    Arguments:
        networks: PyTorch neural network objects
        valid_data: PyTorch data loader for the validation set
        criterions: the loss functions
        baseline_model: boolean to determine if the training is done for the baseline model
    
     Returns:
        gen_loss: A scalar for the average loss for generator function over the validation set
        disc_loss: A scalar for the average loss for discriminator function over the validation set
    """
    total_gen_loss = 0.0
    total_disc_loss = 0.0
    total_epoch = 0

    for net in networks:
        net.eval()
    if baseline_model:
        for i, data in enumerate(valid_data, 0):
            # Get the inputs
            inputs, labels = data

            # Place on GPU
            if use_cuda and cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass
            gen_outputs = networks[0](inputs)
            # Calculate the loss
            gen_loss = criterions[0](None, gen_outputs, labels)
            total_gen_loss += gen_loss.item()
            total_epoch += len(labels)
        gen_loss = float(total_gen_loss) / (total_epoch)
        return gen_loss
    else:
        for i, data in enumerate(valid_data, 0):
            # Get the inputs
            inputs, labels = data
            # Forward pass
            gen_outputs = networks[0](inputs)
            disc_labels_outputs = networks[1](inputs, labels)
            disc_generated_outputs = networks[1](inputs, gen_outputs)
            # Calculate the loss
            gen_loss = criterions[0](disc_generated_outputs, gen_outputs, labels)
            disc_loss = criterions[1](disc_labels_outputs, disc_generated_outputs)
            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()
            total_epoch += len(labels)
        gen_loss = float(total_gen_loss) / (total_epoch)
        disc_loss = float(total_disc_loss) / (total_epoch)
        return gen_loss, disc_loss

def train_model(data_path: str, epoch_num: int=200, batch_size: int=128, gen_lr: float=0.0001, disc_lr: float=0.0001, baseline_model: bool=True, use_cuda: bool=True):
    """
    Train the model

    Arguments:
        epoch_num: the number of iterations in learning process
        batch_size: the size of our mini batches
        gen_lr: the speed of learning in gradient descent for the generator
        disc_lr: the speed of learning in gradient descent for the discriminator
        baseline_model: boolean to determine if the training is done for the baseline model
    """
    name = "baseline" if baseline_model else "satgenerator" 
    # Fixed PyTorch random seed for reproducible result
    manual_seed(0)
    random.seed(0)
    # Load the data
    train_data, valid_data, test_data = load_dataset.load_data(path=data_path, batch_size=batch_size)
    # Set up the model fundamentals for the generator
    gen_net = model.Generator()

    if use_cuda and cuda.is_available():
      gen_net.cuda()
    
    gen_criterion = GeneratorLoss(baseline_model=baseline_model)
    gen_optim = optim.Adam(gen_net.parameters(), lr=gen_lr)
    gen_train_loss = zeros(epoch_num)
    gen_val_loss = zeros(epoch_num)

    networks = [gen_net]
    criterions = [gen_criterion]
    optimizers = [gen_optim]
    loss_data = [gen_train_loss, gen_val_loss]

    # Set the model fundamentals for the discriminator
    if baseline_model == False:
        disc_net = model.Discriminator()
        
        if use_cuda and cuda.is_available():
          disc_net.cuda()
        
        disc_criterion = DiscriminatorLoss()
        disc_optim = optim.Adam(disc_net.parameters(), lr=disc_lr)
        disc_train_loss = zeros(epoch_num)
        disc_val_loss = zeros(epoch_num)

        networks.append(disc_net)
        criterions.append(disc_criterion)
        optimizers.append(disc_optim)
        loss_data.append(disc_train_loss)
        loss_data.append(disc_val_loss)

    # Train
    if baseline_model:
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            total_train_loss = 0.0
            total_epoch = 0
            networks[0].train()
            for i, data in enumerate(train_data, 0):
                # Get the inputs
                inputs, labels = data

                # Place on GPU if available
                if use_cuda and cuda.is_available():
                  inputs = inputs.cuda()
                  labels = labels.cuda()

                # Zero the parameter gradients
                optimizers[0].zero_grad()
                # Forward pass, backward pass, and optimize
                gen_outputs = networks[0](inputs)
                gen_loss = criterions[0](None, gen_outputs, labels)
                gen_loss.backward()
                optimizers[0].step()
                # Calculate loss
                total_train_loss += gen_loss.item()
                total_epoch += len(labels)
            # Save average loss data
            loss_data[0][epoch] = float(total_train_loss) / (total_epoch)
            loss_data[1][epoch] = evaluate(networks, valid_data, criterions, baseline_model, use_cuda)
            print(f"epoch #{epoch}  #####  Baseline Training loss = {loss_data[0][epoch]}  #####  Baseline Validation loss = {loss_data[1][epoch]}")

            # Save the model and csv file of the loss
            path = "model_training/results/model={}_epoch_num={}_batch_size={}gen_lr={}".format(name, epoch, batch_size, gen_lr)
            if epoch == epoch_num:
                savetxt("{}_baseline_train_loss.csv".format(path), loss_data[0])
                savetxt("{}_baseline_valid_loss.csv".format(path), loss_data[1])
            if epoch % 5 == 4 or epoch == epoch_num:
                save(networks[0].state_dict(), path)
    
    else:
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            total_gen_train_loss = 0.0
            total_disc_train_loss = 0.0
            total_epoch = 0
            networks[0].train()
            networks[1].train()
            for i, data in enumerate(train_data, 0):
                # Get the inputs
                inputs, labels = data
                # Zero the parameter gradients
                optimizers[0].zero_grad()
                optimizers[1].zero_grad()
                # Forward pass, backward pass, and optimize
                gen_outputs = networks[0](inputs)
                disc_labels_outputs = networks[1](inputs, labels)
                disc_generated_outputs = networks[1](inputs, gen_outputs)
                
                gen_loss = criterions[0](disc_generated_outputs, gen_outputs, labels)
                disc_loss = criterions[1](disc_labels_outputs, disc_generated_outputs)
                gen_loss.backward()
                disc_loss.backward()
                optimizers[0].step()
                optimizers[1].step()
                # Calculate loss
                total_gen_train_loss += gen_loss.item()
                total_disc_train_loss += disc_loss.item()
                total_epoch += len(labels)
            # Save average loss data
            loss_data[0][epoch] = float(total_gen_train_loss) / (total_epoch)
            loss_data[2][epoch] = float(total_disc_train_loss) / (total_epoch)
            loss_data[1][epoch], loss_data[3][epoch] = evaluate(networks, valid_data, criterions, baseline_model, use_cuda)
            print(f"epoch #{epoch}  #####  Generator Training loss = {loss_data[0][epoch]}  #####  Generator Validation loss = {loss_data[1][epoch]}")
            print(f"epoch #{epoch}  #####  Discriminator Training loss = {loss_data[2][epoch]}  #####  Discriminator Validation loss = {loss_data[3][epoch]}")

            # Save the model and csv file of the loss
            path = "model_training/results/model={}_epoch_num={}_batch_size={}gen_lr={}_disc_lr={}".format(name, epoch, batch_size, gen_lr, disc_lr)
            if epoch == epoch_num:
                savetxt("{}_gen_train_loss.csv".format(path), loss_data[0])
                savetxt("{}_gen_valid_loss.csv".format(path), loss_data[1])
                savetxt("{}_disc_train_loss.csv".format(path), loss_data[2])
                savetxt("{}_disc_valid_loss.csv".format(path), loss_data[3])
            if epoch % 5 == 4 or epoch == epoch_num:
                save(networks[0].state_dict(), path)