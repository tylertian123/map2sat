from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy.random as seed
from sklearn.model_selection import train_test_split

def load_data(path: str, batch_size: int=64, validation_size: float=0.1, test_size: float=0.1, dataloader_test: bool=False):
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
    input_path = path + "/map"
    label_path = path + "/sat"
    transform = T.Compose([T.ToTensor()])

    # Save the images in tensors
    # ImageFolder is used to provide the option to fairly distribute the training, validation, and testing dataset based on location classification
    input_dataset = ImageFolder(input_path, transform=transform)
    label_dataset = ImageFolder(label_path, transform=transform)
    
    # Distribute the data into three sets for both input and label
    seed.seed(0)
    locations = [d[1] for d in input_dataset]
    train_set_inputs, other_set_inputs = train_test_split(input_dataset, test_size=(validation_size+test_size), shuffle=True, stratify=locations)   # other_set is the combination of validation and testing
    other_locations = [d[1] for d in other_set_inputs]
    valid_set_inputs, test_set_inputs = train_test_split(other_set_inputs, test_size=(test_size/(validation_size+test_size)), shuffle=True, stratify=other_locations)

    seed.seed(0)
    locations = [d[1] for d in label_dataset]
    train_set_labels, other_set_labels = train_test_split(label_dataset, test_size=(validation_size+test_size), shuffle=True, stratify=locations)   # other_set is the combination of validation and testing
    other_locations = [d[1] for d in other_set_labels]
    valid_set_labels, test_set_labels = train_test_split(other_set_labels, test_size=(test_size/(validation_size+test_size)), shuffle=True, stratify=other_locations)

    # Redefine the datasets (ie. set both input and label as images)
    train_set = [(train_set_inputs[i][0], train_set_labels[i][0]) for i in range(len(train_set_inputs))]
    valid_set = [(valid_set_inputs[i][0], valid_set_labels[i][0]) for i in range(len(valid_set_inputs))]
    test_set = [(test_set_inputs[i][0], test_set_labels[i][0]) for i in range(len(test_set_inputs))]

    # Load the data of each set into batches
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=1)

    def test_dataloader():
        """
        Testing if the proportion of data in each set is as defined
        Making sure the inputs and labels match with a few examples
        """
        print(f"Size of training dataset: {len(train_set)}\nSize of validation dataset: {len(valid_set)}\nSize of testing dataset: {len(test_set)}")

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