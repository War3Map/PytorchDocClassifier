import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from classifier.datasets import TrainImageDataset
from classifier.utils import get_val_augmentations, get_train_augmentations, preprocess_data
from classifier.classifier_settings import (
    MODEL_WEIGHTS_PATH,
    TRAIN_PATH,
    CLASSES,
    DATASET_CSV_PATH,
    CSV_DELIMITER, IMAGE_SIZE, N_EPOCHS, BATCH_SIZE, NUM_WORKERS,
)


def get_loaders(_train_df, _train_labels, _val_df, _val_labels, _albumentations_transform,
                _albumentations_transform_validate, _batch_size, _num_workers):
    train_data = TrainImageDataset(dataframe=_train_df,
                                   labels=_train_labels,
                                   path=TRAIN_PATH,
                                   transform=_albumentations_transform)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=_batch_size,
                              num_workers=_num_workers,
                              shuffle=True,
                              drop_last=False)

    validate_data = TrainImageDataset(dataframe=_val_df,
                                      labels=_val_labels,
                                      path=TRAIN_PATH,
                                      transform=_albumentations_transform_validate)
    validate_loader = DataLoader(dataset=validate_data,
                                 batch_size=_batch_size,
                                 num_workers=_num_workers,
                                 shuffle=False,
                                 drop_last=False)
    return train_loader, validate_loader


def main():

    device = torch.device("cuda:0")
    train_losses = []
    val_losses = []
    df = pd.read_csv(DATASET_CSV_PATH, delimiter=CSV_DELIMITER)
    df.reset_index(inplace=True)
    # device_ids = [0, 1]

    albumentations_transform = get_train_augmentations(IMAGE_SIZE)
    albumentations_transform_validate = get_val_augmentations(IMAGE_SIZE)

    model = models.resnext50_32x4d(pretrained=True)
    classes_count = len(CLASSES)
    model.fc = nn.Linear(2048, classes_count)
    # model = nn.DataParallel(model, device_ids=device_ids, output_device=device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=200)

    best_acc_val = 0
    best_acc_train = 0
    for epoch in range(N_EPOCHS):
        train_df, val_df, train_labels, val_labels = preprocess_data(df)
        args = [train_df, train_labels, val_df, val_labels, albumentations_transform, albumentations_transform_validate,
                BATCH_SIZE, NUM_WORKERS]
        train_loader, validate_loader = get_loaders(*args)
        train_len = len(train_loader)
        model.train()
        train_loss = 0
        train_acc = 0

        for i, (imgs, labels) in tqdm(enumerate(train_loader), total=train_len):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = torch.argmax(torch.softmax(output, 1), 1).cpu().detach().numpy()
            true = labels.cpu().numpy()
            train_acc += accuracy_score(true, pred)
            scheduler.step(epoch + i / train_len)

        model.eval()
        val_loss = 0
        acc_val = 0
        val_len = len(validate_loader)
        for i, (imgs, labels) in tqdm(enumerate(validate_loader), total=val_len):
            with torch.no_grad():
                imgs_vaild, labels_vaild = imgs.to(device), labels.to(device)
                output_test = model(imgs_vaild)
                val_loss += criterion(output_test, labels_vaild).item()
                pred = torch.argmax(torch.softmax(output_test, 1), 1).cpu().detach().numpy()
                true = labels.cpu().numpy()
                acc_val += accuracy_score(true, pred)

        avg_val_acc = acc_val / val_len
        avg_train_acc = train_acc / train_len

        print(f'Epoch {epoch + 1}/{N_EPOCHS}  train_loss {train_loss / train_len} '
              f'train_acc {train_acc / train_len}  '
              f'val_loss {val_loss / val_len}  val_acc {avg_val_acc}')

        if avg_val_acc > best_acc_val:
            best_acc_val = avg_val_acc
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
        elif (avg_val_acc == best_acc_val) and (avg_train_acc == best_acc_train):
            best_acc_train = avg_train_acc
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
        train_losses.append(train_loss / train_len)
        val_losses.append(val_loss / val_len)
    return train_losses, val_losses


if __name__ == '__main__':
    main()
