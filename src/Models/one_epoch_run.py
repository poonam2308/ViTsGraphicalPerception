import sklearn
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import mean_absolute_error


def trainingEpochWithoutNan(model, data_loader, criterion, optimizer, epoch, device):
    training_loss = 0.0

    for i, batch in enumerate(data_loader):
        x_batch = batch['image'].to(device)
        y_batch = batch['label'].unsqueeze(1).to(device)

        outputs = model(x_batch)

        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        if i % 200 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] trainloss: {training_loss / 200:.3f}')
            training_loss = 0.0
    return training_loss


def trainingEpoch(model, data_loader, criterion, optimizer, epoch, device):
    training_loss = 0.0

    for i, batch in enumerate(data_loader):
        x_batch = batch['image'].to(device)
        y_batch = batch['label'].unsqueeze(1).to(device)

        outputs = model(x_batch)

        loss = criterion(outputs, y_batch)

        if torch.isnan(loss):
            print(f'NaN detected in loss at epoch {epoch + 1}, batch {i + 1}. Skipping this batch.')
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        if i % 200 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] trainloss: {training_loss / 200:.3f}')
            training_loss = 0.0

    return training_loss


# for pretrained network
def trainingEpoch_pre(model, data_loader, criterion, optimizer, epoch, device):
    training_loss = 0.0
    for i, batch in enumerate(data_loader):
        x_batch = batch['image'].to(device)
        y_batch = batch['label'].unsqueeze(1).to(device)

        outputs = model(x_batch).logits

        loss = criterion(outputs, y_batch)

        if torch.isnan(loss):
            print(f'NaN detected in loss at epoch {epoch + 1}, batch {i + 1}. Skipping this batch.')
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        if i % 200 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] trainloss: {training_loss / 200:.3f}')
            training_loss = 0.0

    return training_loss


def trainingEpochWithGradClip(model, data_loader, criterion, optimizer, epoch, device, clip_gradient_norm=None):
    training_loss = 0.0

    for i, batch in enumerate(data_loader):
        x_batch = batch['image'].to(device)
        y_batch = batch['label'].unsqueeze(1).to(device)

        outputs = model(x_batch)

        loss = criterion(outputs, y_batch)

        if torch.isnan(loss):
            print(f'NaN detected in loss at epoch {epoch + 1}, batch {i + 1}. Skipping this batch.')
            continue

        optimizer.zero_grad()
        loss.backward()

        if clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient_norm)

        optimizer.step()

        training_loss += loss.item()
        if i % 200 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] trainloss: {training_loss / 200:.3f}')
            training_loss = 0.0

    return training_loss


def trainingEpochWithGradClip_pre(model, data_loader, criterion, optimizer, epoch, device, clip_gradient_norm=None):
    training_loss = 0.0

    for i, batch in enumerate(data_loader):
        x_batch = batch['image'].to(device)
        y_batch = batch['label'].unsqueeze(1).to(device)

        outputs = model(x_batch).logits

        loss = criterion(outputs, y_batch)

        if torch.isnan(loss):
            print(f'NaN detected in loss at epoch {epoch + 1}, batch {i + 1}. Skipping this batch.')
            continue

        optimizer.zero_grad()
        loss.backward()

        if clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient_norm)

        optimizer.step()

        training_loss += loss.item()
        if i % 200 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] trainloss: {training_loss / 200:.3f}')
            training_loss = 0.0

    return training_loss


def validationEpoch_withoutnan(model, data_loader, criterion, epoch, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            x_batch = batch['image'].to(device)
            y_batch = batch['label'].unsqueeze(1).to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()

            if i % 200 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] valloss: {val_loss / 200:.3f}')
                val_loss = 0.0
    return val_loss


def validationEpoch(model, data_loader, criterion, epoch, device):
    model.eval()
    val_loss = 0.0
    # criterion = nn.MSELoss()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            x_batch = batch['image'].to(device)
            y_batch = batch['label'].unsqueeze(1).to(device)
            if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
                print("NaN detected in input data. Skipping batch.")
                continue

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            if torch.isnan(loss).any():
                print("NaN detected in loss. Skipping batch.")
                continue

            val_loss += loss.item()

            if i % 200 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] valloss: {val_loss / 200:.3f}')
                val_loss = 0.0

    return val_loss


def validationEpoch_pre(model, data_loader, criterion, epoch, device):
    model.eval()
    val_loss = 0.0
    # criterion = nn.MSELoss()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            x_batch = batch['image'].to(device)
            y_batch = batch['label'].unsqueeze(1).to(device)
            if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
                print("NaN detected in input data. Skipping batch.")
                continue

            outputs = model(x_batch).logits
            loss = criterion(outputs, y_batch)

            if torch.isnan(loss).any():
                print("NaN detected in loss. Skipping batch.")
                continue

            val_loss += loss.item()

            if i % 200 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] valloss: {val_loss / 200:.3f}')
                val_loss = 0.0

    return val_loss


def validationEpoch_withEalryStopping(model, data_loader, criterion, epoch, device, patience, min_delta,
                                      epochs_without_improvement, best_loss):
    model.eval()
    val_loss = 0.0
    # criterion = nn.MSELoss()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            x_batch = batch['image'].to(device)
            y_batch = batch['label'].unsqueeze(1).to(device)
            if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
                print("NaN detected in input data. Skipping batch.")
                continue
            outputs = model(x_batch).logits
            loss = criterion(outputs, y_batch)
            if torch.isnan(loss).any():
                print("NaN detected in loss. Skipping batch.")
                continue
            val_loss += loss.item()
            if i % 200 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] val_loss: {val_loss / 200:.3f}')
                val_loss = 0.0

        average_val_loss = val_loss / len(data_loader)
        stop_training, best_loss, epochs_without_improvement = early_stopping(patience, min_delta, average_val_loss,
                                                                              best_loss, epochs_without_improvement)

    return average_val_loss, best_loss, epochs_without_improvement, stop_training


def testingEpochOne1(model, data_loader, device):
    loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch['image'].to(device)
            y_test = batch['label'].unsqueeze(1).to(device)
            y_pred = model(x_batch)

            loss += mean_absolute_error(y_pred.cpu().numpy() * 100, y_test.cpu().numpy() * 100)

    mean_loss = loss / len(data_loader)

    MLAE = np.log2(mean_loss + 0.125)

    return MLAE



def testingEpochOne(model, data_loader, device):
    losses = []

    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch['image'].to(device)
            y_test = batch['label'].unsqueeze(1).to(device)
            y_pred = model(x_batch)

            if torch.isnan(y_pred).any() or torch.isnan(y_test).any():
                print("Warning: NaN values found in predictions or true labels. Skipping this batch.")
                continue

            batch_loss = mean_absolute_error(y_pred.cpu().numpy() * 100, y_test.cpu().numpy() * 100)

            losses.append(batch_loss)

    if np.isnan(losses).any():
        print("Warning: NaN values found in computed losses. Skipping NaN values for mean calculation.")
        losses = [x for x in losses if not np.isnan(x)]

    mean_loss = np.mean(losses)

    MLAE = np.log2(mean_loss + 0.125)

    return MLAE


def testingEpochOneTask(model, data_loader, device):
    losses = []
    all_y_pred = []
    all_y_true = []

    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch['image'].to(device)
            y_test = batch['label'].unsqueeze(1).to(device)
            y_pred = model(x_batch)

            if torch.isnan(y_pred).any() or torch.isnan(y_test).any():
                print("Warning: NaN values found in predictions or true labels. Skipping this batch.")
                continue

            all_y_pred.append(y_pred.cpu())
            all_y_true.append(y_test.cpu())

            batch_loss = mean_absolute_error(y_pred.cpu().numpy() * 100, y_test.cpu().numpy() * 100)
            losses.append(batch_loss)

    if np.isnan(losses).any():
        print("Warning: NaN values found in computed losses. Skipping NaN values for mean calculation.")
        losses = [x for x in losses if not np.isnan(x)]

    mean_loss = np.mean(losses)
    MLAE = np.log2(mean_loss + 0.125)

    y_pred_all = torch.cat(all_y_pred).view(-1).numpy()
    y_true_all = torch.cat(all_y_true).view(-1).numpy()

    return MLAE, y_pred_all, y_true_all


def testingEpoch1(model, data_loader, device):
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch['image'].to(device)
            y_test = batch['label'].unsqueeze(1).to(device)
            y_pred = model(x_batch)

            y_pred_reshaped = y_pred.view(-1, y_pred.shape[-2], y_pred.shape[-1])
            y_test_reshaped = y_test.view(-1, y_test.shape[-2], y_test.shape[-1])

            batch_losses = mean_absolute_error(y_pred_reshaped.cpu().numpy() * 100, y_test_reshaped.cpu().numpy() * 100,
                                               multioutput='raw_values')

            losses.extend(batch_losses)

    mean_loss = np.mean(losses)

    MLAE = np.log2(mean_loss + 0.125)

    return MLAE


def testingEpoch(model, data_loader, device):
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch['image'].to(device)
            y_test = batch['label'].unsqueeze(1).to(device)
            y_pred = model(x_batch)

            assert y_pred.numel() == y_test.numel(), "Number of elements in y_pred and y_test do not match"

            y_pred_flat = y_pred.view(-1)
            y_test_flat = y_test.view(-1)

            batch_loss = mean_absolute_error(y_pred_flat.cpu().numpy() * 100, y_test_flat.cpu().numpy() * 100)

            losses.append(batch_loss)

    mean_loss = np.mean(losses)

    MLAE = np.log2(mean_loss + 0.125)

    return MLAE

def testingEpochTask(model, data_loader, device):
    y_pred_all = []
    y_true_all = []
    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch['image'].to(device)
            y_test = batch['label'].unsqueeze(1).to(device)
            y_pred = model(x_batch)

            y_pred_all.append(y_pred.cpu())
            y_true_all.append(y_test.cpu())

    y_pred_all = torch.cat(y_pred_all, dim=0).view(-1).numpy()
    y_true_all = torch.cat(y_true_all, dim=0).view(-1).numpy()

    mean_loss = mean_absolute_error(y_pred_all * 100, y_true_all * 100)
    MLAE = np.log2(mean_loss + 0.125)

    return MLAE, y_pred_all, y_true_all

def testingEpoch_pre(model, data_loader, device):
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch['image'].to(device)
            y_test = batch['label'].unsqueeze(1).to(device)
            y_pred = model(x_batch).logits

            assert y_pred.numel() == y_test.numel(), "Number of elements in y_pred and y_test do not match"

            y_pred_flat = y_pred.view(-1)
            y_test_flat = y_test.view(-1)

            if torch.isnan(y_pred).any() or torch.isnan(y_test).any():
                print("Warning: NaN values found in predictions or true labels. Skipping this batch.")
                continue

            batch_loss = mean_absolute_error(y_pred_flat.cpu().numpy() * 100, y_test_flat.cpu().numpy() * 100)

            losses.append(batch_loss)

    mean_loss = np.mean(losses)

    MLAE = np.log2(mean_loss + 0.125)

    return MLAE


def testingEpochOne_pre(model, data_loader, device):
    losses = []

    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch['image'].to(device)
            y_test = batch['label'].unsqueeze(1).to(device)
            y_pred = model(x_batch).logits

            if torch.isnan(y_pred).any() or torch.isnan(y_test).any():
                print("Warning: NaN values found in predictions or true labels. Skipping this batch.")
                continue

            batch_loss = mean_absolute_error(y_pred.cpu().numpy() * 100, y_test.cpu().numpy() * 100)

            losses.append(batch_loss)

    if np.isnan(losses).any():
        print("Warning: NaN values found in computed losses. Skipping NaN values for mean calculation.")
        losses = [x for x in losses if not np.isnan(x)]

    mean_loss = np.mean(losses)

    MLAE = np.log2(mean_loss + 0.125)

    return MLAE


def early_stopping(patience, min_delta, current_loss, best_loss, epochs_without_improvement):
    """
    Early stopping based on validation loss.

    Parameters:
    - patience: Number of epochs with no improvement after which training will be stopped
    - min_delta: Minimum change in the monitored quantity to qualify as an improvement
    - current_loss: Loss value for the current epoch
    - best_loss: Best observed validation loss so far
    - epochs_without_improvement: Number of consecutive epochs without improvement

    Returns:
    - stop_training: Boolean indicating whether to stop the training process
    - best_loss: Updated best validation loss
    - epochs_without_improvement: Updated count of consecutive epochs without improvement
    """
    stop_training = False

    # Check if current loss is an improvement
    if current_loss < best_loss - min_delta:
        best_loss = current_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Check early stopping conditions
    if epochs_without_improvement >= patience:
        print(f'Early stopping after {epochs_without_improvement} epochs without improvement.')
        stop_training = True

    return stop_training, best_loss, epochs_without_improvement


#### for deit

def deit_trainingEpoch(model, data_loader, optimizer, epoch, device):
    training_loss = 0.0

    for i, batch in enumerate(data_loader):
        x_batch = batch['image'].to(device)
        y_batch = batch['label'].unsqueeze(1).to(device)
        loss = model(x_batch, y_batch)

        if torch.isnan(loss):
            print(f'NaN detected in loss at epoch {epoch + 1}, batch {i + 1}. Skipping this batch.')
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        if i % 200 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] trainloss: {training_loss / 200:.3f}')
            training_loss = 0.0

    return training_loss

def deit_validationEpoch(v, model, data_loader, epoch, device):
    v.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            x_batch = batch['image'].to(device)
            y_batch = batch['label'].unsqueeze(1).to(device)

            if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
                print("NaN detected in input data. Skipping batch.")
                continue

            loss = model(x_batch, y_batch)

            if torch.isnan(loss).any():
                print("NaN detected in loss. Skipping batch.")
                continue

            val_loss += loss.item()

            if i % 200 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] valloss: {val_loss / 200:.3f}')
                val_loss = 0.0

    return val_loss


