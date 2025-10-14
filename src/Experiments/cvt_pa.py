import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.ClevelandMcGill.figure3 import Figure3
from src.Models.cvt import CvTRegression
from src.Models.one_epoch_run import trainingEpoch, validationEpoch, testingEpoch
from src.Datasets.position_angle_data import pa_data_generation, pa_normalization_data, PositionAngleData
from src.config_utils import get_args_parser

args = get_args_parser()
args = args.parse_args()

FIGURE3 = 'Figure3.'
DATATYPE_LIST = ['data_to_barchart', 'data_to_piechart', 'data_to_piechart_aa']
NOISE = True
# DATA GENERATION
for i in range(len(DATATYPE_LIST)):
    DATATYPE = eval(FIGURE3 + DATATYPE_LIST[i])
    X_train, y_train, X_val, y_val, X_test, y_test = pa_data_generation(DATATYPE, NOISE=True,
                                                                        train_target=args.train_target,
                                                                        val_target=args.val_target,
                                                                        test_target=args.test_target)
    # Normalize Data In-place
    X_train = pa_normalization_data(X_train)
    y_train = pa_normalization_data(y_train)

    X_val = pa_normalization_data(X_val)
    y_val = pa_normalization_data(y_val)

    X_test = pa_normalization_data(X_test)
    y_test = pa_normalization_data(y_test)

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

    train_dataset = PositionAngleData(X_train, y_train, transform=transform, channels=True)
    val_dataset = PositionAngleData(X_val, y_val, transform=transform, channels=True)
    test_dataset = PositionAngleData(X_test, y_test, transform=transform, channels=True)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    # Instantiate the model
    cvt_model = CvTRegression(num_classes=5, channels=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvt_model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(cvt_model.parameters(),  lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum, nesterov=args.nesterov)
    training_loss = []
    validation_loss = []

    for epoch in range(args.epochs):
        train_loss = trainingEpoch(cvt_model, train_loader, criterion, optimizer, epoch, device)
        training_loss.append(train_loss)
        val_loss = validationEpoch(cvt_model, val_loader, criterion, epoch, device)
        validation_loss.append(val_loss)
        torch.save(cvt_model.state_dict(), 'chkpt/cvt3n3_' + DATATYPE_LIST[i] + '.pth')

    plt.figure()
    plt.plot(training_loss, label='training')
    plt.plot(validation_loss, label='validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("training and validation loss")
    plt.savefig('trainingplots/cvt3n3_'+DATATYPE_LIST[i] + '.png')

    MLAE = testingEpoch(cvt_model, test_loader, device)
    print("MLAE", round(MLAE, 2))
