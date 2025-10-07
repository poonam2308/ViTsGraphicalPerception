import os
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.Models.one_epoch_run import trainingEpoch, validationEpoch, testingEpochOneTask
from src.Datasets.perceptiondata import data_generation, normalization_data, PerceptionDataset
from src.Models.swin import SwinRegression


FIGURE1 = 'Figure1.'
DATATYPE_LIST = ['position_common_scale', 'position_non_aligned_scale', 'length', 'direction', 'angle', 'area',
                 'volume', 'curvature', 'shading']
NOISE = True

task_flag_limits = {
    'curvature': 4,
    'length': 4,
    'position_non_aligned_scale': 4,
}

# DATA GENERATION
for i, task in enumerate(DATATYPE_LIST):
    DATATYPE = eval(FIGURE1 + task)
    max_flags = task_flag_limits.get(task, 3)

    for flag_count in range(max_flags):
        FLAGS = [False] * 10
        for f in range(flag_count):
            FLAGS[f] = True

        model_path = f'chkpt/largedata_r/swin1_reg_{task}_flags{flag_count}.pth'
        plot_path = f'trainingplots/largedata_r/swin1_reg_{task}_flags{flag_count}.png'
        stats_path = f'stats/largedata_r/swin1_reg_{task}_flags{flag_count}.pkl'

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)

        X_train, y_train, X_val, y_val, X_test, y_test = data_generation(
            DATATYPE, FLAGS=FLAGS, NOISE=NOISE, train_target=120000,
            val_target=40000, test_target=40000
        )
        # Normalize Data In-place
        X_train = normalization_data(X_train)
        y_train = normalization_data(y_train)

        X_val = normalization_data(X_val)
        y_val = normalization_data(y_val)

        X_test = normalization_data(X_test)
        y_test = normalization_data(y_test)

        X_train -= 0.5
        X_val -= 0.5
        X_test -= 0.5

        print('memory usage', (X_train.element_size() * X_train.nelement() +
                               X_val.element_size() * X_val.nelement() +
                               X_test.element_size() * X_test.nelement() +
                               y_train.element_size() * y_train.nelement() +
                               y_val.element_size() * y_val.nelement() +
                               y_test.element_size() * y_test.nelement()) / (1024 * 1024), 'MB')

        ############################################
        # Define a transformation to convert images into patches (adjust parameters as needed)
        transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])

        train_dataset = PerceptionDataset(X_train, y_train, transform=transform, channels=True)
        val_dataset = PerceptionDataset(X_val, y_val, transform=transform, channels=True)
        test_dataset = PerceptionDataset(X_test, y_test, transform=transform, channels=True)

        train_loader = DataLoader(train_dataset, 16, shuffle=True)
        val_loader = DataLoader(val_dataset, 16, shuffle=True)
        test_loader = DataLoader(test_dataset, 16, shuffle=True)

        # Instantiate the model

        swin_model = SwinRegression(
            hidden_dim=96,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            channels=3,
            num_outputs=1,
            head_dim=32,
            window_size=7,
            downscaling_factors=(4, 2, 2, 2),
            relative_pos_embedding=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        swin_model.to(device)

        criterion = nn.MSELoss()

        optimizer = torch.optim.SGD(swin_model.parameters(), lr=0.0001, weight_decay=1e-6, momentum=0.9, nesterov=True)
        training_loss = []
        validation_loss = []

        for epoch in range(100):
            train_loss = trainingEpoch(swin_model, train_loader, criterion, optimizer, epoch, device)
            training_loss.append(train_loss)
            val_loss = validationEpoch(swin_model, val_loader, criterion, epoch, device)
            validation_loss.append(val_loss)

        # Save model
        torch.save(swin_model.state_dict(), model_path)

        # Save plot
        plt.figure()
        plt.plot(training_loss, label='training')
        plt.plot(validation_loss, label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"{task} | FLAGS={flag_count}")
        plt.savefig(plot_path)
        plt.close()

        # MLAE + prediction
        MLAE, y_pred, y_true = testingEpochOneTask(swin_model, test_loader, device)

        stats = {
            'task': task,
            'flag_count': flag_count,
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'y_pred': y_pred,
            'y_true': y_true,
            'MLAE': MLAE
        }
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        print(f"[{task}] FLAGS={flag_count} MLAE: {round(MLAE, 2)}")

