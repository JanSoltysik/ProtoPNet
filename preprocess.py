import torch
from torchvision.transforms import ToTensor, Compose, Grayscale
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from settings import train_dir, test_dir


def calculate_mean_and_std():
    train_dataset = ImageFolder(
        train_dir,
        transform=Compose([
            # Grayscale(num_output_channels=1),
            ToTensor()
        ])
    )
    loader = DataLoader(train_dataset, batch_size=len(train_dataset), num_workers=1)
    data = next(iter(loader))
    return data[0].mean().item(), data[0].std().item()


mean, std = (0.15810531377792358,), (0.31045639514923096,)
#calculate_mean_and_std()


def preprocess(x, mean, std):
    # return (x - mean) / std
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[0]) / std[0]
    return y


def preprocess_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    return preprocess(x, mean=mean, std=std)


def undo_preprocess(x, mean, std):
    # return x * std + mean
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[0] + mean[0]
    return y


def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    return undo_preprocess(x, mean=mean, std=std)


if __name__ == "__main__":
    print("##########TEST################")
    # test modified functions
    train_dataset = ImageFolder(
        test_dir,
        transform=Compose([
            # Grayscale(num_output_channels=1),
            ToTensor()
        ])
    )
    loader = DataLoader(train_dataset, batch_size=1, num_workers=1)
    data = next(iter(loader))
    print(calculate_mean_and_std())

    # print(data[0].size())
    # print(preprocess(data[0], mean, std))
