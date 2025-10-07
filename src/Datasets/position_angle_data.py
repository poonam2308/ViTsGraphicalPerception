import torch
import numpy as np
import time
from torch.utils.data import Dataset

from src.ClevelandMcGill.figure3 import Figure3

# reference : Run_bar_framed_rectangle_from_scratch
def pa_normalization_data(x):
    x_min = x.min()
    x_max = x.max()
    x -= x_min
    x /= (x_max - x_min)
    return x


def pa_data_generation(DATATYPE, NOISE=False, train_target=60000, val_target=20000, test_target=20000):
    if NOISE == 'True':
        NOISE = True
        SUFFIX = '_noise.'
    else:
        NOISE = False

    train_labels = []
    val_labels = []
    test_labels = []

    X_train = torch.zeros((train_target, 1, 100, 100), dtype=torch.float32)
    y_train = torch.zeros((train_target, 5), dtype=torch.float32)
    train_counter = 0

    X_val = torch.zeros((val_target, 1, 100, 100), dtype=torch.float32)
    y_val = torch.zeros((val_target, 5), dtype=torch.float32)
    val_counter = 0

    X_test = torch.zeros((test_target, 1, 100, 100), dtype=torch.float32)
    y_test = torch.zeros((test_target, 5), dtype=torch.float32)
    test_counter = 0

    t0 = time.time()

    all_counter = 0

    while train_counter < train_target or val_counter < val_target or test_counter < test_target:
        all_counter += 1
        data, label = Figure3.generate_datapoint()

        pot = np.random.choice(3)
        # sometimes we know which pot is right
        if label in train_labels:
            pot = 0
        if label in val_labels:
            pot = 1
        if label in test_labels:
            pot = 2

        if pot == 0 and train_counter < train_target:

            if label not in train_labels:
                train_labels.append(label)

            image = DATATYPE(data)
            image = image.astype(np.float32)

            if NOISE:
                image += np.random.uniform(0, 0.05, (100, 100))
            X_train[train_counter] = torch.from_numpy(image)
            y_train[train_counter] = torch.tensor(label, dtype=torch.float32)
            train_counter += 1

        elif pot == 1 and val_counter < val_target:

            if label not in val_labels:
                val_labels.append(label)

            image = DATATYPE(data)
            image = image.astype(np.float32)

            if NOISE:
                image += np.random.uniform(0, 0.05, (100, 100))

            X_val[val_counter] = torch.from_numpy(image)
            y_val[val_counter] = torch.tensor(label, dtype=torch.float32)
            val_counter += 1

        elif pot == 2 and test_counter < test_target:
            if label not in test_labels:
                test_labels.append(label)

            image = DATATYPE(data)
            image = image.astype(np.float32)

            if NOISE:
                image += np.random.uniform(0, 0.05, (100, 100))
            X_test[test_counter] = torch.from_numpy(image)
            y_test[test_counter] = torch.tensor(label, dtype=torch.float32)
            test_counter += 1

    print('Done', time.time() - t0, 'seconds (', all_counter, 'iterations)')
    return X_train, y_train, X_val, y_val, X_test, y_test


# custom dataset using pytorch
class PositionAngleData(Dataset):
    def __init__(self, images, labels, transform=None, channels=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.channels = channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.channels:
            img = img.expand(3, *img.shape[1:])

        if not torch.is_tensor(img):
            img = torch.from_numpy(img)

        if self.transform:
            img = self.transform(img)

        return {'image': img, 'label': label}
