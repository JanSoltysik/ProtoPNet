base_architecture = 'vgg19'
img_size = 56
prototype_shape = (6, 128, 4, 4)
num_classes = 2
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001'

data_path = '/mnt/users/jsoltysik/local/ro/MIL/'
train_dir = data_path + 'train_augmented/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {
    'features': 1e-4,
    'add_on_layers': 3e-3,
    'prototype_vectors': 3e-3
}

joint_lr_step_size = 5

warm_optimizer_lrs = {
    'add_on_layers': 3e-3,
    'prototype_vectors': 3e-3
}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 6
num_warm_epochs = 1

push_start = 5
push_epochs = [i for i in range(num_train_epochs) if i % 5 == 0]
