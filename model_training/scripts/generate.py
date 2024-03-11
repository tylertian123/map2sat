import torch
import numpy as np
from matplotlib import gridspec, pyplot as plt

import itertools

from model_training.scripts.model import Generator, Discriminator
from model_training.scripts.train import Hyperparameters
from model_training.scripts.load_dataset import load_data


def load_model(hp: Hyperparameters, type: str = None, device: str = "cpu") -> Generator | Discriminator:
    """
    Load a saved model.
    
    device can be "cpu" or "gpu", which will remap the model onto a different device.
    If None, the device the model was originally on will be used.

    The type argument should only be specified for non-baseline models, in which case 
    it should be either "gen" or "disc", specifying whether to load a generator or 
    discriminator.
    """

    path = hp.get_path("models")
    if type is not None:
        path += "_" + type
        if type == "gen":
            model = Generator()
        else:
            model = Discriminator()
    else:
        model = Generator()

    if device == "cpu":
        map_location = "cpu"
    elif device == "gpu":
        map_location = "cuda:0"
    else:
        map_location = None
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model

def img_tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    """
    Convert a pytorch tensor to a numpy array in the format that matplotlib expects.
    """
    return img.detach().numpy().transpose((1, 2, 0))

def plot_outputs(hp: Hyperparameters, path: str, subset: str, n: int, start: int = 0):
    """
    Plot generated outputs alongside inputs and ground truth.

    path is the dataset directory. subset is either "train", "validation" or "test".
    n samples will be generated and displayed. If start is nonezero, the first samples in the dataset will be skipped.
    """
    model = load_model(hp, device="cpu")
    model.eval()
    train_loader, val_loader, test_loader = load_data(path, batch_size=1, dataloader_test=False)
    if subset == "train":
        loader = train_loader
    elif subset == "validation":
        loader = val_loader
    elif subset == "test":
        loader = test_loader
    else:
        raise ValueError(subset)
    
    plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(n, 3)
    gs.update(wspace=0.025, hspace=0.05)

    for i, (inputs, labels) in enumerate(itertools.islice(loader, start, None)):
        output = img_tensor_to_numpy(model(inputs)[0])
        input = img_tensor_to_numpy(inputs[0])
        label = img_tensor_to_numpy(labels[0])

        for j, img in zip(range(3), (input, output, label)):
            ax = plt.subplot(gs[i * 3 + j])
            ax.imshow(img)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        if i + 1 >= n:
            break
