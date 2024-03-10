from train import train_model, evaluate
from load_dataset import load_data
from model import Generator, Discriminator

if __name__ == "__main__":
    train_model(data_path="data/")