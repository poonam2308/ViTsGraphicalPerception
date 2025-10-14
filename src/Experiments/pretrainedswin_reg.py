import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import SwinForImageClassification
from src.ClevelandMcGill.figure1 import Figure1
from src.Models.one_epoch_run import trainingEpoch_pre, validationEpoch_pre, testingEpochOne_pre
from src.Datasets.perceptiondata import data_generation, normalization_data, PerceptionDataset
from src.config_utils import get_args_parser, init_wandb

args = get_args_parser()
args = args.parse_args()

init_wandb(args, tags=args.wandb_tags, group=args.group)

FIGURE1 = 'Figure1.'
DATATYPE_LIST = ['position_common_scale', 'position_non_aligned_scale', 'length', 'direction', 'angle', 'area',
                 'volume', 'curvature', 'shading']

# DATA GENERATION
for i in range(len(DATATYPE_LIST)):
    DATATYPE = eval(FIGURE1+DATATYPE_LIST[i])
    X_train, y_train, X_val, y_val, X_test, y_test = data_generation(DATATYPE, NOISE=args.NOISE,
                                                                     train_target=args.train_target,
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
    # Define a transformation to convert images into patches (adjust parameters as needed)
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
    swin_model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    swin_model.classifier = torch.nn.Linear(swin_model.config.hidden_size, 1)  # REGRESSION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    swin_model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(swin_model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum, nesterov=args.nesterov)

    training_loss = []
    validation_loss = []

    for epoch in range(args.epochs):  # Adjust the number of epochs as needed
        train_loss = trainingEpoch_pre(swin_model, train_loader, criterion, optimizer, epoch, device)
        training_loss.append(train_loss)
        val_loss = validationEpoch_pre(swin_model, val_loader, criterion, epoch, device)
        validation_loss.append(val_loss)
        torch.save(swin_model.state_dict(), 'chkpt/pretrained_swin_' + DATATYPE_LIST[i] + '.pth')

    plt.figure()
    plt.plot(training_loss, label='training')
    plt.plot(validation_loss, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("training and validation loss")
    plt.savefig('trainingplots/pretrained_swin_'+DATATYPE_LIST[i] + '.png')
    plt.legend()

    MLAE = testingEpochOne_pre(swin_model, test_loader, device)

    print("MLAE", round(MLAE, 2))
