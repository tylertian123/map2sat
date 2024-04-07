from train import train_model, evaluate, Hyperparameters
from load_dataset import load_data
from model import Generator, Discriminator

if __name__ == "__main__":
    #cont = Hyperparameters(name="leaky-250", baseline_model=False, gen_lr=0.0001, disc_lr=0.0001, epoch_num=179, batch_size=64)
    #cont = Hyperparameters(name="leaky-250", baseline_model=False, gen_lr=0.00002, disc_lr=0.00002, epoch_num=229, batch_size=64)
    cont = Hyperparameters(name="leaky-250", baseline_model=False, gen_lr=0.00001, disc_lr=0.00001, epoch_num=269, batch_size=64)
    hp = cont.copy_and_change(epoch_num=300)
    train_model(hp=hp, cont_hp=cont)