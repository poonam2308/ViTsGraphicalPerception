import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
import wandb
from src.ClevelandMcGill.figure4 import Figure4
from src.config_utils import get_args_parser, init_wandb
from src.Models.one_epoch_run import trainingEpoch_pre, validationEpoch_pre, testingEpoch_pre
from src.Datasets.position_length_data import pl_data_generation, pl_normalization_data, PositionLengthData

args = get_args_parser()
args = args.parse_args()
init_wandb(args, tags=args.wandb_tags, group=args.group)

FIGURE4 = 'Figure4.'
DATATYPE_LIST = ['data_to_type1', 'data_to_type2', 'data_to_type3', 'data_to_type4', 'data_to_type5']
NOISE = True
# DATA GENERATION
for i in range(len(DATATYPE_LIST)):
    DATATYPE = eval(FIGURE4 + DATATYPE_LIST[i])
    X_train, y_train, X_val, y_val, X_test, y_test = pl_data_generation(DATATYPE, NOISE=args.NOISE,
                                                                        train_target=args.train_target,
                                                                        val_target=args.val_target,
                                                                        test_target=args.test_target)
    # Normalize Data In-place
    X_train = pl_normalization_data(X_train)
    y_train = pl_normalization_data(y_train)

    X_val = pl_normalization_data(X_val)
    y_val = pl_normalization_data(y_val)

    X_test = pl_normalization_data(X_test)
    y_test = pl_normalization_data(y_test)

    X_train -= 0.5
    X_val -= 0.5
    X_test -= 0.5

    print('memory usage',
          (X_train.nbytes + X_val.nbytes + X_test.nbytes + y_train.nbytes + y_val.nbytes + y_test.nbytes) / 1000000.,
          'MB')

    ############################################
    # Define a transformation to convert images into patches (adjust parameters as needed)
    transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    train_dataset = PositionLengthData(X_train, y_train, transform=transform, channels=True)
    val_dataset = PositionLengthData(X_val, y_val, transform=transform, channels=True)
    test_dataset = PositionLengthData(X_test, y_test, transform=transform, channels=True)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    # Instantiate the model
    model_name = "google/vit-base-patch16-224"
    vit_model = ViTForImageClassification.from_pretrained(model_name)

    vit_model.classifier = torch.nn.Linear(vit_model.config.hidden_size, 5)  # REGRESSION num_classes =5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(vit_model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum, nesterov=args.nesterov)
    training_loss = []
    validation_loss = []
    for epoch in range(args.epochs):
        train_loss = trainingEpoch_pre(vit_model, train_loader, criterion, optimizer, epoch, device)
        training_loss.append(train_loss)
        val_loss = validationEpoch_pre(vit_model, val_loader, criterion, epoch, device)
        validation_loss.append(val_loss)
        torch.save(vit_model.state_dict(), 'chkpt/pretrained_pl_vit_' + DATATYPE_LIST[i] + '.pth')

        wandb.log({"train/loss": train_loss})
        wandb.log({"val/loss": val_loss})

    plt.figure()
    plt.plot(training_loss, label='training')
    plt.plot(validation_loss, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("training and validation loss")
    plt.savefig('trainingplots/pretrained_pl_vit_' + DATATYPE_LIST[i] + '.png')
    plt.legend()
    MLAE = testingEpoch_pre(vit_model, test_loader, device)
    print("MLAE", round(MLAE, 2))
