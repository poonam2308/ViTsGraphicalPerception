import torch
import numpy as np
import time
from torch.utils.data import Dataset
from src.ClevelandMcGill.figure1 import Figure1
from src.ClevelandMcGill.figure12 import Figure12
from src.ClevelandMcGill.figure3 import Figure3
from src.ClevelandMcGill.figure4 import Figure4
from src.ClevelandMcGill.weber import Weber

np.random.seed(0)
torch.manual_seed(0)


# Reference : run_regression_from_scratch_multi
def test_normalization_data(x):
    x_min = x.min()
    x_max = x.max()
    x -= x_min
    x /= (x_max - x_min)
    return x


def test_reg_data_generation(DATATYPE, FLAGS=None, NOISE=False, test_target=20000):
    if FLAGS is None:
        FLAGS = [False] * 10

    X_test = torch.zeros((test_target, 1, 100, 100), dtype=torch.float32)
    y_test = torch.zeros(test_target, dtype=torch.float32)

    test_counter = 0
    t0 = time.time()
    all_counter = 0

    while test_counter < test_target:
        all_counter += 1
        sparse, image, label, parameters = DATATYPE(FLAGS)
        image = image.astype(np.float32)

        if NOISE:
            image += np.random.uniform(0, 0.05, (100, 100))
        X_test[test_counter] = torch.from_numpy(image)
        y_test[test_counter] = label
        test_counter += 1

    print('Done', time.time() - t0, 'seconds (', all_counter, 'iterations)')
    return X_test, y_test


def test_bfr_data_generation(DATATYPE, FLAGS=None, NOISE=False, test_target=20000):
    if FLAGS is None:
        FLAGS = [False] * 10

    X_test = torch.zeros((test_target, 1, 100, 100), dtype=torch.float32)
    y_test = torch.zeros((test_target, 2), dtype=torch.float32)
    test_counter = 0
    t0 = time.time()
    all_counter = 0

    while test_counter < test_target:
        all_counter += 1
        data, label, parameters = Figure12.generate_datapoint()
        image = DATATYPE(data)
        image = image.astype(np.float32)

        if NOISE:
            image += np.random.uniform(0, 0.05, (100, 100))
        X_test[test_counter] = torch.from_numpy(image)
        y_test[test_counter] = torch.tensor(label, dtype=torch.float32)
        test_counter += 1

    print('Done', time.time() - t0, 'seconds (', all_counter, 'iterations)')
    return X_test, y_test


def test_pa_data_generation(DATATYPE, FLAGS=None, NOISE=False, test_target=20000):
    if FLAGS is None:
        FLAGS = [False] * 10

    X_test = torch.zeros((test_target, 1, 100, 100), dtype=torch.float32)
    y_test = torch.zeros((test_target, 5), dtype=torch.float32)
    test_counter = 0
    t0 = time.time()
    all_counter = 0

    while test_counter < test_target:
        all_counter += 1
        data, label = Figure3.generate_datapoint()

        image = DATATYPE(data)
        image = image.astype(np.float32)

        if NOISE:
            image += np.random.uniform(0, 0.05, (100, 100))
        X_test[test_counter] = torch.from_numpy(image)
        y_test[test_counter] = torch.tensor(label, dtype=torch.float32)
        test_counter += 1

    print('Done', time.time() - t0, 'seconds (', all_counter, 'iterations)')
    return X_test, y_test


def test_pl_data_generation(DATATYPE, FLAGS=None, NOISE=False, test_target=20000):
    if FLAGS is None:
        FLAGS = [False] * 10

    X_test = torch.zeros((test_target, 1, 100, 100), dtype=torch.float32)
    y_test = torch.zeros((test_target, 5), dtype=torch.float32)
    test_counter = 0
    t0 = time.time()
    all_counter = 0

    while test_counter < test_target:
        all_counter += 1
        data, label = Figure4.generate_datapoint()

        image = DATATYPE(data)
        image = image.astype(np.float32)

        if NOISE:
            image += np.random.uniform(0, 0.05, (100, 100))
        X_test[test_counter] = torch.from_numpy(image)
        y_test[test_counter] = torch.tensor(label, dtype=torch.float32)
        test_counter += 1

    print('Done', time.time() - t0, 'seconds (', all_counter, 'iterations)')
    return X_test, y_test


def test_wb_data_generation(DATATYPE, FLAGS=None, NOISE=False, test_target=20000):
    if FLAGS is None:
        FLAGS = [False] * 10

    X_test = torch.zeros((test_target, 1, 100, 100), dtype=torch.float32)
    y_test = torch.zeros(test_target, dtype=torch.float32)
    test_counter = 0
    t0 = time.time()
    all_counter = 0

    while test_counter < test_target:
        all_counter += 1
        image, label = DATATYPE()
        image = image.astype(np.float32)

        if NOISE:
            image += np.random.uniform(0, 0.05, (100, 100))
        X_test[test_counter] = torch.from_numpy(image)
        y_test[test_counter] = torch.tensor(label, dtype=torch.float32)
        test_counter += 1

    print('Done', time.time() - t0, 'seconds (', all_counter, 'iterations)')
    return X_test, y_test


# custom dataset using pytorch
class TestDataset(Dataset):
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
