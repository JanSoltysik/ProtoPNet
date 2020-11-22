import Augmentor
import os
from settings import data_path


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


datasets_root_dir = data_path
dir = datasets_root_dir + 'train/'
target_dir = datasets_root_dir + 'train_augmented/'
train_size = 60000

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]
    # rotation
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    # p.flip_left_right(probability=0.5)
    # p.flip_left_right(probability=0.5)
    # for i in range(10):
    p.sample(train_size)

    # skew
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    # p.flip_left_right(probability=0.5)
    # for i in range(10):
    p.sample(train_size)

    # shear
    # p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    # p.flip_left_right(probability=0.5)
    # for i in range(10):
    p.sample(train_size)
