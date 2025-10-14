import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.ClevelandMcGill.weber import Weber
from src.Models.one_epoch_run import trainingEpoch, validationEpoch, testingEpochOne
from src.Models.vit import ViTRegression
from src.Datasets.weber_data import wb_data_generation, wb_normalization_data, WeberData
from src.config_utils import get_args_parser

args = get_args_parser()
args = args.parse_args()
WEBER = 'Weber.'
DATATYPE_LIST = ['base10', 'base100', 'base1000']
NOISE = True
# DATA GENERATION
for i in range(len(DATATYPE_LIST)):
    DATATYPE = eval(WEBER + DATATYPE_LIST[i])
    X_train, y_train, X_val, y_val, X_test, y_test = wb_data_generation(DATATYPE, NOISE=True, train_target=60000,
                                                                        val_target=20000, test_target=20000)
    # Normalize Data In-place
    X_train = wb_normalization_data(X_train)
    y_train = wb_normalization_data(y_train)

    X_val = wb_normalization_data(X_val)
    y_val = wb_normalization_data(y_val)

    X_test = wb_normalization_data(X_test)
    y_test = wb_normalization_data(y_test)

    X_train -= 0.5
    X_val -= 0.5
    X_test -= 0.5

    print('memory usage',
          (X_train.nbytes + X_val.nbytes + X_test.nbytes + y_train.nbytes + y_val.nbytes + y_test.nbytes) / 1000000.,
          'MB')

    ############################################
    transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    train_dataset = WeberData(X_train, y_train, transform=transform, channels=True)
    val_dataset = WeberData(X_val, y_val, transform=transform, channels=True)
    test_dataset = WeberData(X_test, y_test, transform=transform, channels=True)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    vit_model = ViTRegression(image_size=(224, 224), patch_size=(16, 16), num_classes=1, dim=512, depth=8,
                              heads=4, mlp_dim=1024, channels=3)
    #change for the tiny vit
    # vit_model = ViTRegression(image_size=(224, 224), patch_size=(8, 8), num_classes=1, dim=192, depth=12,
    #                           heads=3, mlp_dim=768, channels=3)


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
        torch.save(vit_model.state_dict(), 'chkpt/vit3wn_' + DATATYPE_LIST[i] + '.pth')

    plt.figure()
    plt.plot(training_loss, label='training')
    plt.plot(validation_loss, label='validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("training and validation loss")
    plt.savefig('trainingplots/vit3wn_' + DATATYPE_LIST[i] + '.png')
    MLAE = testingEpochOne(vit_model, test_loader, device)
    print("MLAE", round(MLAE, 2))
