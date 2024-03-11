from model_training.scripts.load_dataset import load_data
import model_training.scripts.model as model

import dataclasses

import torch
from torch import nn, manual_seed, ones_like, zeros_like, optim, save, cuda
from torch.utils.data import DataLoader
from numpy import random, zeros, savetxt
from tqdm.auto import tqdm

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
            gan_loss = self.sig_ce(disc_generator_result, real_grid)   # the generator should be good enought that the gan classifies it as real
        l1_loss = self.mae(generated_image, label)   # reconstruction error

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

        generated_loss = self.sig_ce(disc_generator_result, fake_grid)   # the discriminator has to classify the generated image as fake
        real_loss = self.sig_ce(disc_label_result, real_grid)   # the discriminator has to classify the label image as real

        total_disc_loss = generated_loss + real_loss

        return total_disc_loss

@dataclasses.dataclass(frozen=True)
class Hyperparameters:
    """
    Class aggregating hyperparameter settings for a model.

    Note: To prevent bugs, the contents of an instance are NOT modifiable. Treat it like a tuple.
    Use copy_and_change() to get a copy with some fields changed.
    """

    epoch_num: int = 20
    batch_size: int = 64
    gen_lr: float = 0.0001
    disc_lr: float = 0.0001
    baseline_model: bool = True

    def get_filename(self):
        name = "baseline" if self.baseline_model else "satgenerator"
        filename = f"model={name}-epoch_num={self.epoch_num}-batch_size={self.batch_size}-gen_lr={self.gen_lr}"
        if not self.baseline_model:
            filename = f"{filename}-disc_lr={self.disc_lr}"
        return filename
    
    def get_path(self, dir_type: str):
        return f"model_training/{dir_type}/{self.get_filename()}"

    def copy_and_change(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

@torch.no_grad()
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
    total_samples = 0

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
            total_gen_loss += gen_loss.item() * len(labels)
            total_samples += len(labels)
        gen_loss = float(total_gen_loss) / (total_samples)
        return gen_loss
    else:
        for i, data in enumerate(valid_data, 0):
            # Get the inputs
            inputs, labels = data
            # Place on GPU
            if use_cuda and cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # Forward pass
            gen_outputs = networks[0](inputs)
            disc_labels_outputs = networks[1](inputs, labels)
            disc_generated_outputs = networks[1](inputs, gen_outputs)
            # Calculate the loss
            gen_loss = criterions[0](disc_generated_outputs, gen_outputs, labels)
            disc_loss = criterions[1](disc_labels_outputs, disc_generated_outputs)
            total_gen_loss += gen_loss.item() * len(labels)
            total_disc_loss += disc_loss.item() * len(labels)
            total_samples += len(labels)
        gen_loss = float(total_gen_loss) / (total_samples)
        disc_loss = float(total_disc_loss) / (total_samples)
        return gen_loss, disc_loss

def train_model(data_path: str, hp: Hyperparameters, use_cuda: bool=True):
    """
    Train the model

    Arguments:
        epoch_num: the number of iterations in learning process
        batch_size: the size of our mini batches
        gen_lr: the speed of learning in gradient descent for the generator
        disc_lr: the speed of learning in gradient descent for the discriminator
        baseline_model: boolean to determine if the training is done for the baseline model
    """
    # Fixed PyTorch random seed for reproducible result
    manual_seed(0)
    random.seed(0)
    # Load the data
    train_data, valid_data, _ = load_data(path=data_path, batch_size=hp.batch_size)
    # Set up the model fundamentals for the generator
    gen_net = model.Generator()

    if use_cuda and cuda.is_available():
        gen_net.cuda()
    
    gen_criterion = GeneratorLoss(baseline_model=hp.baseline_model)
    gen_optim = optim.Adam(gen_net.parameters(), lr=hp.gen_lr)
    gen_train_loss = zeros(hp.epoch_num)
    gen_val_loss = zeros(hp.epoch_num)

    networks = [gen_net]
    criterions = [gen_criterion]
    optimizers = [gen_optim]
    loss_data = [gen_train_loss, gen_val_loss]

    print("INFO: Generator has", sum(p.numel() for p in networks[0].parameters()), "weights")

    # Set the model fundamentals for the discriminator
    if not hp.baseline_model:
        disc_net = model.Discriminator()
        
        if use_cuda and cuda.is_available():
            disc_net.cuda()
        
        disc_criterion = DiscriminatorLoss()
        disc_optim = optim.Adam(disc_net.parameters(), lr=hp.disc_lr)
        disc_train_loss = zeros(hp.epoch_num)
        disc_val_loss = zeros(hp.epoch_num)

        networks.append(disc_net)
        criterions.append(disc_criterion)
        optimizers.append(disc_optim)
        loss_data.append(disc_train_loss)
        loss_data.append(disc_val_loss)

        print("INFO: Discriminator has", sum(p.numel() for p in networks[1].parameters()), "weights")

    # Train
    if hp.baseline_model:
        print("================== Baseline training starting ==================")
        for epoch in tqdm(range(hp.epoch_num), desc="Epochs"):  # loop over the dataset multiple times
            total_train_loss = 0.0 # Scaled by the number of samples
            total_samples = 0
            networks[0].train()
            for i, data in tqdm(enumerate(train_data, 0), desc="Iterations", total=len(train_data), leave=False):
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
                total_train_loss += gen_loss.item() * len(labels)
                total_samples += len(labels)

            # Save average loss data
            loss_data[0][epoch] = float(total_train_loss) / (total_samples)
            loss_data[1][epoch] = evaluate(networks, valid_data, criterions, hp.baseline_model, use_cuda)
            tqdm.write(f"EPOCH #{epoch}  #####  Baseline Training loss = {loss_data[0][epoch]}  #####  Baseline Validation loss = {loss_data[1][epoch]}")

            # Save the model and csv file of the loss
            if epoch == hp.epoch_num - 1:
                path = hp.copy_and_change(epoch_num=epoch).get_path("results")
                savetxt("{}_baseline_train_loss.csv".format(path), loss_data[0])
                savetxt("{}_baseline_valid_loss.csv".format(path), loss_data[1])
            if epoch % 5 == 4 or epoch == hp.epoch_num - 1:
                path = hp.copy_and_change(epoch_num=epoch).get_path("models")
                save(networks[0].state_dict(), path)
    
    else:
        print("================== Model training starting ==================")
        for epoch in tqdm(range(hp.epoch_num), desc="Epochs"):  # loop over the dataset multiple times
            total_gen_train_loss = 0.0 # Scaled by number of samples
            total_disc_train_loss = 0.0
            total_samples = 0
            networks[0].train()
            networks[1].train()
            for i, data in tqdm(enumerate(train_data, 0), desc="Iterations", total=len(train_data), leave=False):
                # Get the inputs
                inputs, labels = data
                # Place on GPU if available
                if use_cuda and cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                
                # Train the discriminator
                optimizers[1].zero_grad()
                disc_labels_outputs = networks[1](inputs, labels) # Discriminator output on real inputs
                gen_outputs = networks[0](inputs) # Generator outputs
                # Detach the generator outputs so gradients are not calculated for generator weights
                disc_generated_outputs = networks[1](inputs, gen_outputs.detach()) # Discriminator output on fake inputs
                disc_loss = criterions[1](disc_labels_outputs, disc_generated_outputs)
                disc_loss.backward()
                optimizers[1].step()

                # Train the generator
                optimizers[0].zero_grad()
                # Recompute discriminator outputs on the generator outputs, this time without detach
                disc_generated_outputs = networks[1](inputs, gen_outputs)
                gen_loss = criterions[0](disc_generated_outputs, gen_outputs, labels)
                gen_loss.backward()
                optimizers[0].step()
                
                # Calculate loss
                total_gen_train_loss += gen_loss.item() * len(labels)
                total_disc_train_loss += disc_loss.item() * len(labels)
                total_samples += len(labels)
                
            # Save average loss data
            loss_data[0][epoch] = float(total_gen_train_loss) / (total_samples)
            loss_data[2][epoch] = float(total_disc_train_loss) / (total_samples)
            loss_data[1][epoch], loss_data[3][epoch] = evaluate(networks, valid_data, criterions, hp.baseline_model, use_cuda)
            tqdm.write(f"EPOCH #{epoch}  #####  Generator Training loss = {loss_data[0][epoch]}  #####  Generator Validation loss = {loss_data[1][epoch]}")
            tqdm.write(f"EPOCH #{epoch}  #####  Discriminator Training loss = {loss_data[2][epoch]}  #####  Discriminator Validation loss = {loss_data[3][epoch]}")

            # Save the model and csv file of the loss
            if epoch == hp.epoch_num - 1:
                path = hp.copy_and_change(epoch_num=epoch).get_path("results")
                savetxt("{}_gen_train_loss.csv".format(path), loss_data[0])
                savetxt("{}_gen_valid_loss.csv".format(path), loss_data[1])
                savetxt("{}_disc_train_loss.csv".format(path), loss_data[2])
                savetxt("{}_disc_valid_loss.csv".format(path), loss_data[3])
            if epoch % 5 == 4 or epoch == hp.epoch_num - 1:
                path = hp.copy_and_change(epoch_num=epoch).get_path("models")
                save(networks[0].state_dict(), path)
