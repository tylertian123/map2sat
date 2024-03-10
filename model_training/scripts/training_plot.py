import matplotlib.pyplot as plt
import pandas as pd

def plot_csv(split: str, name: str, epoch: int, batch_size: int, lr: float):
    """ Sample usage: plot_csv("validation", name="baseline", epoch=99, batch_size=64, lr=0.0001)"""

    split = split.lower()
    split_to_abbrev = {"training": "train", "validation": "valid", "testing": "test"}
    split_abbrev = split_to_abbrev[split] if split in split_to_abbrev.keys() else split

    filepath = f"model_training/results/model={name}-epoch_num={epoch}-batch_size={batch_size}-gen_lr={lr}_{name}_{split_abbrev}_loss.csv"

    df = pd.read_csv(filepath, header=None, names=['Loss'])

    plt.plot(df['Loss'])
    plt.title(f'{split} loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Show the plot
    plt.show()