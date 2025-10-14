import os
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.ClevelandMcGill.figure1 import Figure1
from src.Models.one_epoch_run import trainingEpoch, validationEpoch, testingEpochTask
from src.Datasets.perceptiondata import data_generation, normalization_data, PerceptionDataset
from src.Models.vit import ViTRegression
from src.config_utils import get_args_parser

args = get_args_parser()
args = args.parse_args()
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
    max_flags = task_flag_limits.get(task, 3)  # Default to 0-2 if not listed

    for flag_count in range(max_flags):
        FLAGS = [False] * 10
        for f in range(flag_count):
            FLAGS[f] = True

        model_path = f'chkpt/vit_reg_{task}_flags{flag_count}.pth'
        plot_path = f'trainingplots/vit_reg_{task}_flags{flag_count}.png'
        stats_path = f'stats/vit_reg_{task}_flags{flag_count}.pkl'

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)


        X_train, y_train, X_val, y_val, X_test, y_test = data_generation(
            DATATYPE, FLAGS=FLAGS, NOISE=NOISE,  train_target=args.train_target,
            val_target=args.val_target,
            test_target=args.test_target)
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
        transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])

        train_dataset = PerceptionDataset(X_train, y_train, transform=transform, channels=True)
        val_dataset = PerceptionDataset(X_val, y_val, transform=transform, channels=True)
        test_dataset = PerceptionDataset(X_test, y_test, transform=transform, channels=True)

        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        # Instantiate the model
        vit_model = ViTRegression(image_size=(224, 224), patch_size=(8, 8), num_classes=1, dim=192, depth=12,
                                  heads=3, mlp_dim=768, channels=3)

        # model = ViTRegression(
        #     image_size=(224, 224),
        #     patch_size=(16, 16),
        #     dim=512,  # You can adjust the overall model dimension as well
        #     depth=8,  # You might also consider adjusting the depth of the model
        #     heads=4,  # Experiment with a lower number of heads
        #     mlp_dim=1024  # Experiment with a lower MLP dimension
        # )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vit_model.to(device)

        criterion = nn.MSELoss()

        optimizer = torch.optim.SGD(vit_model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum, nesterov=args.nesterov)
        training_loss = []
        validation_loss = []

        for epoch in range(args.epochs):
            train_loss = trainingEpoch(vit_model, train_loader, criterion, optimizer, epoch, device)
            training_loss.append(train_loss)
            val_loss = validationEpoch(vit_model, val_loader, criterion, epoch, device)
            validation_loss.append(val_loss)

        torch.save(vit_model.state_dict(), model_path)

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
        MLAE, y_pred, y_true = testingEpochTask(vit_model, test_loader, device)

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
