import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ImagesTestDS

from classifier_settings import (
    MODEL_WEIGHTS_PATH,
    TEST_DATASET_PATH,
    MODE,
    CLASSES,
    RESULT_FILE,
    RESULT_DIR,
    HOME_DIR,
    PREDICT_DEVICE,
    RESULT_COLUMNS, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
)
from utils import get_val_augmentations


def main():
    device = PREDICT_DEVICE
    albumentations_transform_validate = get_val_augmentations(IMAGE_SIZE)

    annotations_list = []
    for root, dirs, files in os.walk(TEST_DATASET_PATH):
        if len(files) == 0:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            label = 0
            annotations_list.append({
                'label': label,
                'filepath': filepath,
            })

    annotations_list_new = []
    for annotation in annotations_list:
        annotations_list_new.append({
            'label': annotation['label'],
            'filepath': annotation['filepath'].replace('\\', '/'),
        })

    validate_data = ImagesTestDS(annotations_list=annotations_list,
                                 transform=albumentations_transform_validate,
                                 mode=MODE)
    validate_loader = DataLoader(dataset=validate_data,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False,
                                 drop_last=False)

    print(f'test: {len(annotations_list)}')

    model = models.resnext50_32x4d(pretrained=False)
    classes_count = len(CLASSES)
    model.fc = nn.Linear(2048, classes_count)
    # model = nn.DataParallel(model, device_ids=device_ids, output_device=device)
    checkpoint = torch.load(MODEL_WEIGHTS_PATH)
    model.load_state_dict(checkpoint)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    classifier = torch.nn.Softmax()
    # создаём папки с классами
    for class_name in CLASSES:
        os.makedirs(fr'{os.path.join(HOME_DIR, RESULT_DIR, class_name)}', exist_ok=True)

    class_mapping = {idx: cls for idx, cls in enumerate(CLASSES)}

    model.eval()

    classified_list = []
    val_len = len(validate_loader)
    for idx, (imgs, filepaths) in tqdm(enumerate(validate_loader), total=val_len):
        with torch.no_grad():
            imgs = imgs.to(device)
            output_test = model(imgs)
            preds = torch.argmax(classifier(output_test), dim=1)
            # vals = classifier(output_test)[:, preds]
            predicted_classes = [x.item() for x in preds]
            for path, pred in zip(filepaths, predicted_classes):
                classified_list.append(
                    (path, pred)
                )

    print(classified_list)
    print("Сохранение результатов ...")
    pd.DataFrame(classified_list, columns=RESULT_COLUMNS).to_csv(RESULT_FILE, sep=';')
    print("Сохранение завершено!")


if __name__ == '__main__':
    main()
