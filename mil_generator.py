import os
import torch
import time
from typing import List, Tuple
from PIL.Image import Image, new
from torch.utils.data import Subset
from torchvision.datasets import MNIST

mnist_dir = "/mnt/users/jsoltysik/local/ro/mnist"
mil_dir = "/mnt/users/jsoltysik/local/ro/MIL_69"
target_labels = (6, 9)  # target label == 1 if '9' and '6' is in the bag
bag_size = 4


def concat_pictures(images: Subset) -> Tuple[Image, int]:
    assert len(images) == bag_size

    width = images[0][0].width
    height = images[0][0].height

    bag = new('L', (2 * width, 2 * height))

    bag.paste(images[0][0], (0, 0))
    bag.paste(images[1][0], (width, 0))
    bag.paste(images[2][0], (0, height))
    bag.paste(images[3][0], (width, height))

    # is target_label in the created bag
    _, labels = zip(*images)
    label = int(all(target_label in labels for target_label in target_labels))

    return bag, label


def bag_generator(data, target_dir):
    # generate len(data) of bags
    for i in range(len(data)):
        print(i, end="\r")
        indices = torch.randint(0, len(data), (bag_size,))
        bag, label = concat_pictures(Subset(data, indices))

        # save bag in dir corresponding to label
        bag.save("{}/{}/{}/{:05d}.jpg".format(mil_dir, target_dir, label, i))


def mil_generator():
    train_data = MNIST(mnist_dir, download=True, train=True)
    test_data = MNIST(mnist_dir, download=True, train=False)

    for data, target_dir in zip((train_data, test_data), ('train', 'test')):
        # create required directories if they don't exist
        os.makedirs(f"{mil_dir}/{target_dir}", exist_ok=True)
        os.makedirs(f"{mil_dir}/{target_dir}/0", exist_ok=True)
        os.makedirs(f"{mil_dir}/{target_dir}/1", exist_ok=True)

        # generate bags
        bag_generator(data, target_dir)


if __name__ == "__main__":
    start = time.time()
    mil_generator()
    end = time.time()
    print(f"Generating took {end - start} seconds")

