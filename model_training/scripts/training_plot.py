from model_training.scripts.train import Hyperparameters

import matplotlib.pyplot as plt
import pandas as pd

def plot_csv(split: str, name: str, hp: Hyperparameters):
    """ Sample usage: plot_csv("validation", "baseline", hp)"""

    split = split.lower()
    split_to_abbrev = {"training": "train", "validation": "valid", "testing": "test"}
    split_abbrev = split_to_abbrev[split] if split in split_to_abbrev.keys() else split

    filepath = f"{hp.get_path('results')}_{name}_{split_abbrev}_loss.csv"

    df = pd.read_csv(filepath, header=None, names=['Loss'])

    plt.plot(df['Loss'])
    plt.title(f'{split} loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Show the plot
    plt.show()