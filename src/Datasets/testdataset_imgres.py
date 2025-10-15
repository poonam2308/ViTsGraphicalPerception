# src/Datasets/testdataset_imgres.py

import torch
import torch.nn.functional as F
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

# ---------------------------------------
# Utilities
# ---------------------------------------
def test_normalization_data(x: torch.Tensor):
    x_min = x.min()
    x_max = x.max()
    x -= x_min
    x /= (x_max - x_min + 1e-8)
    return x

def _maybe_resize_np(image_np: np.ndarray, image_res: int | None):
    """Resize a HxW numpy image to (image_res, image_res) if requested."""
    if image_res is None:
        return image_np
    h, w = image_np.shape
    if (h, w) == (image_res, image_res):
        return image_np
    t = torch.from_numpy(image_np)[None, None, ...].float()   # [1,1,H,W]
    t = F.interpolate(t, size=(image_res, image_res), mode="bilinear", align_corners=False)
    return t[0, 0].numpy().astype(np.float32)

def _alloc(test_target: int, label_shape, res: int):
    X = torch.zeros((test_target, 1, res, res), dtype=torch.float32)
    if isinstance(label_shape, int):
        y = torch.zeros(test_target, dtype=torch.float32)
    else:
        y = torch.zeros((test_target,) + tuple(label_shape), dtype=torch.float32)
    return X, y

def _noise(image: np.ndarray, NOISE: bool):
    if NOISE:
        image = image + np.random.uniform(0, 0.05, image.shape).astype(np.float32)
    return image

def _timer_print(t0: float, iters: int, verbose: bool):
    if verbose:
        print('Done', time.time() - t0, 'seconds (', iters, 'iterations)')

# ---------------------------------------
# Generators (now resolution-agnostic)
# Defaults: test_target=100, image_res=224, verbose=False
# ---------------------------------------
def test_reg_data_generation(DATATYPE, FLAGS=None, NOISE=False, test_target=100, image_res: int | None = 224, verbose=False):
    if FLAGS is None:
        FLAGS = [False] * 10

    # prime first sample to infer size (and to resize if requested)
    _, image, label, _ = DATATYPE(FLAGS)
    image = _maybe_resize_np(image.astype(np.float32), image_res)
    res = image.shape[0]
    X_test, y_test = _alloc(test_target, 1, res)

    t0 = time.time()
    iters = 0
    for i in range(test_target):
        if i > 0:
            _, image, label, _ = DATATYPE(FLAGS)
            image = _maybe_resize_np(image.astype(np.float32), image_res)
        image = _noise(image, NOISE)
        X_test[i] = torch.from_numpy(image)
        y_test[i] = label
        iters += 1

    _timer_print(t0, iters, verbose)
    return X_test, y_test

def test_bfr_data_generation(DATATYPE, FLAGS=None, NOISE=False, test_target=100, image_res: int | None = 224, verbose=False):
    if FLAGS is None:
        FLAGS = [False] * 10

    data, label, _ = Figure12.generate_datapoint()
    image = _maybe_resize_np(DATATYPE(data).astype(np.float32), image_res)
    res = image.shape[0]
    X_test, y_test = _alloc(test_target, (2,), res)

    t0 = time.time()
    iters = 0
    for i in range(test_target):
        if i > 0:
            data, label, _ = Figure12.generate_datapoint()
            image = _maybe_resize_np(DATATYPE(data).astype(np.float32), image_res)
        image = _noise(image, NOISE)
        X_test[i] = torch.from_numpy(image)
        y_test[i] = torch.tensor(label, dtype=torch.float32)
        iters += 1

    _timer_print(t0, iters, verbose)
    return X_test, y_test

def test_pa_data_generation(DATATYPE, FLAGS=None, NOISE=False, test_target=100, image_res: int | None = 224, verbose=False):
    if FLAGS is None:
        FLAGS = [False] * 10

    data, label = Figure3.generate_datapoint()
    image = _maybe_resize_np(DATATYPE(data).astype(np.float32), image_res)
    res = image.shape[0]
    X_test, y_test = _alloc(test_target, (5,), res)

    t0 = time.time()
    iters = 0
    for i in range(test_target):
        if i > 0:
            data, label = Figure3.generate_datapoint()
            image = _maybe_resize_np(DATATYPE(data).astype(np.float32), image_res)
        image = _noise(image, NOISE)
        X_test[i] = torch.from_numpy(image)
        y_test[i] = torch.tensor(label, dtype=torch.float32)
        iters += 1

    _timer_print(t0, iters, verbose)
    return X_test, y_test

def test_pl_data_generation(DATATYPE, FLAGS=None, NOISE=False, test_target=100, image_res: int | None = 224, verbose=False):
    if FLAGS is None:
        FLAGS = [False] * 10

    data, label = Figure4.generate_datapoint()
    image = _maybe_resize_np(DATATYPE(data).astype(np.float32), image_res)
    res = image.shape[0]
    X_test, y_test = _alloc(test_target, (5,), res)

    t0 = time.time()
    iters = 0
    for i in range(test_target):
        if i > 0:
            data, label = Figure4.generate_datapoint()
            image = _maybe_resize_np(DATATYPE(data).astype(np.float32), image_res)
        image = _noise(image, NOISE)
        X_test[i] = torch.from_numpy(image)
        y_test[i] = torch.tensor(label, dtype=torch.float32)
        iters += 1

    _timer_print(t0, iters, verbose)
    return X_test, y_test

def test_wb_data_generation(DATATYPE, FLAGS=None, NOISE=False, test_target=100, image_res: int | None = 224, verbose=False):
    if FLAGS is None:
        FLAGS = [False] * 10

    image, label = DATATYPE()
    image = _maybe_resize_np(image.astype(np.float32), image_res)
    res = image.shape[0]
    X_test, y_test = _alloc(test_target, 1, res)

    t0 = time.time()
    iters = 0
    for i in range(test_target):
        if i > 0:
            image, label = DATATYPE()
            image = _maybe_resize_np(image.astype(np.float32), image_res)
        image = _noise(image, NOISE)
        X_test[i] = torch.from_numpy(image)
        y_test[i] = torch.tensor(label, dtype=torch.float32)
        iters += 1

    _timer_print(t0, iters, verbose)
    return X_test, y_test

# ---------------------------------------
# Dataset wrapper (unchanged API)
# ---------------------------------------
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
