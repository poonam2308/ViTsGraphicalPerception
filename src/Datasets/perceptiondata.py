import torch
import numpy as np
import time
from torch.utils.data import Dataset
from ClevelandMcGill.figure1 import Figure1

# Reference : run_regression_from_scratch_multi
def normalization_data(x):
    x_min = x.min()
    x_max = x.max()
    x -= x_min
    x /= (x_max - x_min)
    return x


def data_generation(DATATYPE, FLAGS=None, NOISE = False, train_target=60000, val_target=20000, test_target=20000):

    if FLAGS is None:
        FLAGS = [False] * 10
    global_min = np.inf
    global_max = -np.inf

    for N in range(train_target + val_target + test_target):
        sparse, image, label, parameters = DATATYPE(FLAGS)
        global_min = np.min([label, global_min])
        global_max = np.max([label, global_min])

    X_train = torch.zeros((train_target, 1, 100, 100), dtype=torch.float32)
    y_train = torch.zeros(train_target, dtype=torch.float32)
    train_counter = 0

    X_val = torch.zeros((val_target, 1, 100, 100), dtype=torch.float32)
    y_val = torch.zeros(val_target, dtype=torch.float32)
    val_counter = 0

    X_test = torch.zeros((test_target, 1, 100, 100), dtype=torch.float32)
    y_test = torch.zeros(test_target, dtype=torch.float32)
    test_counter = 0

    t0 = time.time()

    min_label = np.inf
    max_label = -np.inf

    all_counter = 0
    while train_counter < train_target or val_counter < val_target or test_counter < test_target:
        all_counter += 1
        sparse, image, label, parameters = DATATYPE(FLAGS)
        image = image.astype(np.float32)

        pot = np.random.choice(3)
        if label == global_min or label == global_max:
            pot = 0

        if pot == 0 and train_counter < train_target:
            if label in y_val or label in y_test:
                continue
            if NOISE:
                image += np.random.uniform(0, 0.05, (100, 100))
            X_train[train_counter] = torch.from_numpy(image)
            y_train[train_counter] = label
            train_counter += 1
        elif pot == 1 and val_counter < val_target:
            if label in y_train or label in y_test:
                continue
            if NOISE:
                image += np.random.uniform(0, 0.05, (100, 100))
            X_val[val_counter] = torch.from_numpy(image)
            y_val[val_counter] = label
            val_counter += 1
        elif pot == 2 and test_counter < test_target:
            if label in y_train or label in y_val:
                continue
            if NOISE:
                image += np.random.uniform(0, 0.05, (100, 100))
            X_test[test_counter] = torch.from_numpy(image)
            y_test[test_counter] = label
            test_counter += 1

    print('Done', time.time() - t0, 'seconds (', all_counter, 'iterations)')
    return X_train, y_train, X_val, y_val, X_test, y_test


# custom dataset using pytorch
class PerceptionDataset(Dataset):
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
            img= img.expand(3,*img.shape[1:])

        if not torch.is_tensor(img):
            img = torch.from_numpy(img)

        if self.transform:
            img= self.transform(img)

        return {'image':img, 'label': label}

#
# DATATYPE = eval('Figure1.position_common_scale')
# NOISE = True
# # DATA GENERATION
# X_train, y_train, X_val, y_val, X_test, y_test = data_generation(DATATYPE, NOISE = True)
# # Normalize Data In-place
# X_train = normalization_data(X_train)
# y_train = normalization_data(y_train)
#
# X_val = normalization_data(X_val)
# y_val = normalization_data(y_val)
#
# X_test = normalization_data(X_test)
# y_test = normalization_data(y_test)
#
# X_train -= 0.5
# X_val -= 0.5
# X_test -= 0.5
#
# print('memory usage', (X_train.element_size() * X_train.nelement() +
#                       X_val.element_size() * X_val.nelement() +
#                       X_test.element_size() * X_test.nelement() +
#                       y_train.element_size() * y_train.nelement() +
#                       y_val.element_size() * y_val.nelement() +
#                       y_test.element_size() * y_test.nelement()) / (1024 * 1024), 'MB')


############################################
# Define a transformation to convert images into patches (adjust parameters as needed)
# transform = transforms.Compose([
#     transforms.Resize((224, 224))
# ])
#
# train_dataset = PerceptionDataset(X_train, y_train, transform=transform)
#
# batch_size = 64
#
# train_loader = DataLoader(train_dataset, 64, shuffle =True)
# # Iterate through the data loader in your training loop
# for batch in train_loader:
#     # Access the input and target tensors
#     x_batch = batch['image']
#     y_batch = batch['label']
#
#     # Your training code here
#     print("Batch input tensor shape:", x_batch.shape)
#     print("Batch target tensor shape:", y_batch.shape)
#     break  # For illustration purposes, breaking after the first batch