from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
import matplotlib.pyplot as plt
import numpy.random as seed
from sklearn.model_selection import train_test_split
import os

def make_dataset(root: str) -> list:
    """Reads a directory with data.
    Returns a dataset as a list of tuples of paired image paths: (input_path, label_path)
    """
    dataset = []

    # Our dir names for inputs and labels
    input_dir = os.path.join(root, 'map')
    label_dir = os.path.join(root, 'sat')

    input_fnames = []
    for folder in os.listdir(input_dir):
        # input_fnames.extend(os.listdir(os.path.join(input_dir, folder)))
        for file in os.listdir(os.path.join(input_dir, folder)):
            input_path = os.path.join(input_dir, folder, file)
            label_file = "sat" + file[3:]
            label_path = os.path.join(label_dir, folder, label_file)
            dataset.append((input_path, label_path))

    return dataset

class CustomVisionDataset(VisionDataset):

    def __init__(self,
                 root,
                 loader=default_loader,
                 input_transform=None,
                 label_transform=None):
        super().__init__(root,
                         transform=input_transform,
                         target_transform=label_transform)

        # Prepare dataset
        samples = make_dataset(self.root)

        self.loader = loader
        self.samples = samples
        # list of input images
        self.input_samples = [s[1] for s in samples]
        # list of label images
        self.label_samples = [s[1] for s in samples]

    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        # getting our paths to images
        input_path, label_path = self.samples[index]

        # import each image using loader (by default it's PIL)
        input_sample = self.loader(input_path)
        label_sample = self.loader(label_path)

        # here goes tranforms if needed
        # maybe we need different tranforms for each type of image
        if self.transform is not None:
            input_sample = self.transform(input_sample)
        if self.target_transform is not None:
            label_sample = self.target_transform(label_sample)      

        # now we return the right imported pair of images (tensors)
        return input_sample, label_sample

    def __len__(self):
        return len(self.samples)


def load_data(path: str, batch_size: int=64, validation_size: float=0.1, test_size: float=0.1, dataloader_test: bool=True):
    """
    Load the data and splitting it into three sections of training, validating, and testing

    Arguments:
        path: the path to the root of the images
        batch_size: the number of samples in each batch
        validation_size: percentage of data used for validation (27% --> 0.27)
        test_size: percentage of data used for testing (27% --> 0.27)
        dataloader_test: boolean to test the dataloader correctness

    Return:
        train_loader: batches of dataset for training
        valid_loader: batches of dataset for validation
        test_loader: batches of dataset for testing
    """
    # input_path = path + "/map"
    # label_path = path + "/sat"
    transform = T.Compose([T.ToTensor()])

    random_seed = 42

    # Save the images in tensors
    # ImageFolder is used to provide the option to fairly distribute the training, validation, and testing dataset based on location classification

    dataset = CustomVisionDataset(path, input_transform=transform, label_transform=transform)
    print(dataset)

    seed.seed(random_seed)
    all_indices = list(range(len(dataset)))
    train_valid, test = train_test_split(all_indices, test_size=test_size, train_size=(1-test_size))
    train, valid = train_test_split(train_valid, test_size=validation_size, train_size=(1-validation_size))

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train, num_workers=1)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid, num_workers=1)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test, num_workers=1)

    # OLDER STRATEGY
    
    # input_dataset = ImageFolder(input_path, transform=transform)
    # label_dataset = ImageFolder(label_path, transform=transform)

    # # Distribute the data into three sets for both input and label
    # seed.seed(random_seed)
    # all_indices = list(range(len(input_dataset)))
    # train_valid, test = train_test_split(all_indices, test_size=test_size, stratify=input_dataset.targets)
    # train, valid = train_test_split(train_valid, test_size=validation_size, stratify=[input_dataset.targets[i] for i in train_valid])

    # train_input_loader = DataLoader(input_dataset, batch_size=batch_size, sampler=train, num_workers=1)
    # valid_input_loader = DataLoader(input_dataset, batch_size=batch_size, sampler=valid, num_workers=1)
    # test_input_loader = DataLoader(input_dataset, batch_size=batch_size, sampler=test, num_workers=1)

    # train_label_loader = DataLoader(label_dataset, batch_size=batch_size, sampler=train, num_workers=1)
    # valid_label_loader = DataLoader(label_dataset, batch_size=batch_size, sampler=valid, num_workers=1)
    # test_label_loader = DataLoader(label_dataset, batch_size=batch_size, sampler=test, num_workers=1)

    # EVEN OLDER STRATEGIES

    # train_set_inputs, temp_inputs = train_test_split(input_dataset, test_size=test_size, random_state=random_seed)
    # test_set_inputs, valid_set_inputs = train_test_split(temp_inputs, test_size=validation_size/(validation_size + test_size), random_state=random_seed)
    # locations = [d[1] for d in input_dataset]
    # train_set_inputs, other_set_inputs = train_test_split(input_dataset, test_size=(validation_size+test_size), shuffle=True, stratify=locations)   # other_set is the combination of validation and testing
    # other_locations = [d[1] for d in other_set_inputs]
    # valid_set_inputs, test_set_inputs = train_test_split(other_set_inputs, test_size=(test_size/(validation_size+test_size)), shuffle=True, stratify=other_locations)

    # seed.seed(random_seed)
    # train_set_labels, temp_labels = train_test_split(label_dataset, test_size=test_size, random_state=random_seed)
    # test_set_labels, valid_set_labels = train_test_split(temp_labels, test_size=validation_size/(validation_size + test_size), random_state=random_seed)
    # locations = [d[1] for d in label_dataset]
    # train_set_labels, other_set_labels = train_test_split(label_dataset, test_size=(validation_size+test_size), shuffle=True, stratify=locations)   # other_set is the combination of validation and testing
    # other_locations = [d[1] for d in other_set_labels]
    # valid_set_labels, test_set_labels = train_test_split(other_set_labels, test_size=(test_size/(validation_size+test_size)), shuffle=True, stratify=other_locations)

    # Redefine the datasets (ie. set both input and label as images)
    # train_set = [(train_set_inputs[i][0], train_set_labels[i][0]) for i in range(len(train_set_inputs))]
    # valid_set = [(valid_set_inputs[i][0], valid_set_labels[i][0]) for i in range(len(valid_set_inputs))]
    # test_set = [(test_set_inputs[i][0], test_set_labels[i][0]) for i in range(len(test_set_inputs))]

    # Load the data of each set into batches
    # train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1)
    # valid_loader = DataLoader(valid_set, S=batch_size, num_workers=1)
    # test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=1)

    def test_dataloader():
        """
        Testing if the proportion of data in each set is as defined
        Making sure the inputs and labels match with a few examples
        """
        # print(f"Size of training dataset: {len(train_set)}\nSize of validation dataset: {len(valid_set)}\nSize of testing dataset: {len(test_set)}")

        # Plot images from all three datasets to make sure the input and label match
        # Each row represents a different dataset, from which 3 pairs of (input, label) are depicted
        _, grid = plt.subplots(3,6)
        # Show the pairs from training set
        data = next(iter(train_loader))
        input, label = data
        for i in range(3):
            grid[0, 2*i].imshow(T.ToPILImage()(input[i]))
            grid[0, 2*i+1].imshow(T.ToPILImage()(label[i]))
        
        # Show the pairs from validation set
        data = next(iter(valid_loader))
        input, label = data
        for i in range(3):
            grid[1, 2*i].imshow(T.ToPILImage()(input[i]))
            grid[1, 2*i+1].imshow(T.ToPILImage()(label[i]))
        
        # Show the pairs from testing set
        data = next(iter(test_loader))
        input, label = data
        for i in range(3):
            grid[2, 2*i].imshow(T.ToPILImage()(input[i]))
            grid[2, 2*i+1].imshow(T.ToPILImage()(label[i]))
        
        plt.show()
    
    if dataloader_test:
        test_dataloader()   # test our data loader
    
    return train_loader, valid_loader, test_loader