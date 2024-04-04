import os
if "COLAB_GPU" in os.environ:
    from model_training.scripts.train import Hyperparameters
else:
    from train import Hyperparameters

import matplotlib.pyplot as plt
import pandas as pd

def plot_csv(split: str, name: str, hp: Hyperparameters, show: bool = True):
    """ Sample usage: plot_csv("validation", "baseline", hp)"""

    split = split.lower()
    split_to_abbrev = {"training": "train", "validation": "valid", "testing": "test"}
    split_abbrev = split_to_abbrev[split] if split in split_to_abbrev.keys() else split

    filepath = f"{hp.get_path('results')}_{name}_{split_abbrev}_loss.csv"

    df = pd.read_csv(filepath, header=None, names=['Loss'])

    plt.plot(df['Loss'][:hp.epoch_num + 1])
    plt.title(f'{split} loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Show the plot
    if show:
        plt.show()

def plot_all(hp: Hyperparameters):
    """Plot training and validation loss for generator and discriminator."""
    if hp.baseline_model:
        plot_csv("training", "baseline", hp, show=False)
        plot_csv("validation", "baseline", hp, show=False)
        plt.legend(["Training", "Validation"])
        plt.title("Training and Validation Loss")
        plt.show()
    else:
        plt.subplot(1, 2, 1)
        plot_csv("training", "gen", hp, show=False)
        plot_csv("validation", "gen", hp, show=False)
        plt.legend(["Training", "Validation"])
        plt.title("Generator Training and Validation Loss")
        plt.subplot(1, 2, 2)
        plot_csv("training", "disc", hp, show=False)
        plot_csv("validation", "disc", hp, show=False)
        plt.legend(["Training", "Validation"])
        plt.title("Discriminator Training and Validation Loss")
        plt.show()
