# Example for testing model! DOESN'T WORK WITH CUSTOM DS!
# Example using Imagewoof

import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from classifier.datasets import Imagewoof
from nn_classifier.classifier_settings import CLASSES
from classifier.utils import get_val_augmentations, preprocess_data


def main():
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    IMAGE_SIZE = 256
    device = torch.device("cuda:0")
    #device_ids = [0, 1]

    albumentations_transform_validate = get_val_augmentations(IMAGE_SIZE)
    train_df, val_df, train_labels, val_labels = preprocess_data('input/noisy_imagewoof.csv')

    validate_data = Imagewoof(dataframe=val_df,
                              labels=val_labels,
                              path='input',
                              transform=albumentations_transform_validate)
    validate_loader = DataLoader(dataset=validate_data,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False,
                                 drop_last=False)

    model = models.resnext50_32x4d(pretrained=False)
    classes_count = len(CLASSES)
    model.fc = nn.Linear(2048, classes_count)
    #model = nn.DataParallel(model, device_ids=device_ids, output_device=device)
    checkpoint = torch.load('model_saved/weight_best.pth')
    model.load_state_dict(checkpoint)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    val_loss = 0
    acc_val = 0
    val_len = len(validate_loader)
    for i, (imgs, labels) in tqdm(enumerate(validate_loader), total=val_len):
        with torch.no_grad():
            imgs_vaild = imgs.to(device)
            labels_vaild = labels.to(device)
            output_test = model(imgs_vaild)
            val_loss += criterion(output_test, labels_vaild).item()
            pred = torch.argmax(torch.softmax(output_test, 1), 1).cpu().detach().numpy()
            true = labels.cpu().numpy()
            acc_val += accuracy_score(true, pred)

    avg_val_acc = acc_val / val_len

    print(f'val_loss {val_loss / val_len}  val_acc {avg_val_acc}')


if __name__ == '__main__':
    main()
