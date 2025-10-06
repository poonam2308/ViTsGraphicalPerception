import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from one_epoch_run import trainingEpoch, validationEpoch, testingEpoch
from position_length_data import pl_data_generation, pl_normalization_data, PositionLengthData
from src.Models.swin import SwinRegression

FIGURE4 = 'Figure4.'
DATATYPE_LIST = ['data_to_type1', 'data_to_type2', 'data_to_type3','data_to_type4','data_to_type5']
NOISE = True
# DATA GENERATION
for i in range(len(DATATYPE_LIST)):
    DATATYPE = eval(FIGURE4 + DATATYPE_LIST[i])
    X_train, y_train, X_val, y_val, X_test, y_test = pl_data_generation(DATATYPE, NOISE=True, train_target=120000,
                                                                        val_target=40000, test_target=40000)
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
    test_dataset = PositionLengthData(X_test, y_test, transform=transform, channels =True)

    train_loader = DataLoader(train_dataset, 8, shuffle=True)
    val_loader = DataLoader(val_dataset, 8, shuffle=True)
    test_loader = DataLoader(test_dataset, 8, shuffle=True)

    # Instantiate the model

    swin_model = SwinRegression(
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        channels=3,
        num_outputs=5,
        head_dim=32,
        window_size=7,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True
    )

    # Move the model to the GPU
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
        torch.save(swin_model.state_dict(), 'chkpt/largedata_r/swin1_' + DATATYPE_LIST[i] + '.pth')

    plt.figure()
    plt.plot(training_loss, label='training')
    plt.plot(validation_loss, label='validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("training and validation loss")
    plt.savefig('trainingplots/largedata_r/swin1_'+DATATYPE_LIST[i] + '.png')

    MLAE = testingEpoch(swin_model, test_loader, device)
    print("MLAE", round(MLAE, 2))
