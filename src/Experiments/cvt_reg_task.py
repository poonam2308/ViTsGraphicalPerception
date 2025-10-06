import os
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from one_epoch_run import validationEpoch, trainingEpochWithGradClip, testingEpochTask
from perceptiondata import data_generation, normalization_data, PerceptionDataset
from src.Models.cvt import CvTRegression


FIGURE1 = 'Figure1.'
DATATYPE_LIST = [
    'position_common_scale', 'position_non_aligned_scale', 'length',
    'direction', 'angle', 'area', 'volume', 'curvature', 'shading'
]

NOISE = True

task_flag_limits = {
    'curvature': 4,
    'length': 4,
    'position_non_aligned_scale': 4,
}

for i, task in enumerate(DATATYPE_LIST):
    DATATYPE = eval(FIGURE1 + task)
    max_flags = task_flag_limits.get(task, 3)  # Default to 0-2 if not listed

    for flag_count in range(max_flags):
        FLAGS = [False] * 10
        for f in range(flag_count):
            FLAGS[f] = True

        model_path = f'chkpt/varied/cvt_reg_{task}_flags{flag_count}.pth'
        plot_path = f'trainingplots/varied/cvt_reg_{task}_flags{flag_count}.png'
        stats_path = f'stats/varied/cvt_reg_{task}_flags{flag_count}.pkl'

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)

        X_train, y_train, X_val, y_val, X_test, y_test = data_generation(
            DATATYPE, FLAGS=FLAGS, NOISE=NOISE, train_target=60000,
            val_target=20000, test_target=20000
        )

        # Normalize
        X_train = normalization_data(X_train)
        y_train = normalization_data(y_train)
        X_val = normalization_data(X_val)
        y_val = normalization_data(y_val)
        X_test = normalization_data(X_test)
        y_test = normalization_data(y_test)

        X_train -= 0.5
        X_val -= 0.5
        X_test -= 0.5

        # Datasets
        transform = transforms.Compose([transforms.Resize((224, 224))])
        train_dataset = PerceptionDataset(X_train, y_train, transform=transform, channels=True)
        val_dataset = PerceptionDataset(X_val, y_val, transform=transform, channels=True)
        test_dataset = PerceptionDataset(X_test, y_test, transform=transform, channels=True)

        train_loader = DataLoader(train_dataset, 32, shuffle=True)
        val_loader = DataLoader(val_dataset, 32, shuffle=True)
        test_loader = DataLoader(test_dataset, 32, shuffle=True)

        # Model setup
        cvt_model = CvTRegression(num_classes=1, channels=3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cvt_model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(cvt_model.parameters(), lr=0.0001, weight_decay=1e-6, momentum=0.9, nesterov=True)

        training_loss = []
        validation_loss = []

        for epoch in range(100):
            clip_gradient_norm = 1.0
            train_loss = trainingEpochWithGradClip(
                cvt_model, train_loader, criterion, optimizer, epoch, device, clip_gradient_norm
            )
            training_loss.append(train_loss)
            val_loss = validationEpoch(cvt_model, val_loader, criterion, epoch, device)
            validation_loss.append(val_loss)

        # Save model
        torch.save(cvt_model.state_dict(), model_path)

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
        MLAE, y_pred, y_true = testingEpochTask(cvt_model, test_loader, device)

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
