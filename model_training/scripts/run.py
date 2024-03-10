from train import train_model, evaluate, Hyperparameters
from load_dataset import load_data
from model import Generator, Discriminator

if __name__ == "__main__":
    hp = Hyperparameters()
    train_model(data_path="data/", hp=hp)