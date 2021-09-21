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
    CLASSES,
    RESULT_FILE,
    RESULT_DIR,
    PREDICT_DEVICE,
    RESULT_COLUMNS, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
)
from utils import get_val_augmentations


def gen_cuda_device_names():
    """
    Generate available devices names
    :return: list of names
    """
    names = ["cpu"]
    cuda_name = "cuda:{}"

    for idx in range(0, 21):
        names.append(cuda_name.format(idx))

    return names


AVAILABLE_DEVICES = gen_cuda_device_names()


def classify(dataset_path, weights_path, result_path,
             result_dir, batch_size, workers_num,
             predicted_classes, device, result_columns,
             need_copy=False):
    """
    Classify images placed by dataset path
    :param workers_num: Number of workers
    :param batch_size: Batch size
    :param result_dir: Dir for saving results
    :param need_copy: True allows to copy source image to result dir
    :param dataset_path: Path to source dataset
    :param weights_path: Path to saved model for classifier
    :param result_path:  Path to save results
    :param predicted_classes: Classes for predictions
    :param device: Device used for classify
    :param result_columns: Columns for result report
    :return:
    """

    running_device = device if device in AVAILABLE_DEVICES else "cpu"
    albumentations_transform_validate = get_val_augmentations(IMAGE_SIZE)

    annotations_list = []
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            label = 0
            annotations_list.append({
                'label': label,
                'filepath': filepath,
            })

    # annotations_list_new = []
    # for annotation in annotations_list:
    #     annotations_list_new.append({
    #         'label': annotation['label'],
    #         'filepath': annotation['filepath'].replace('\\', '/'),
    #     })

    classify_data = ImagesTestDS(annotations_list=annotations_list,
                                 transform=albumentations_transform_validate,
                                 mode='test')
    classify_loader = DataLoader(dataset=classify_data,
                                 batch_size=batch_size,
                                 num_workers=workers_num,
                                 shuffle=False,
                                 drop_last=False)

    print(f'Длина датасета: {len(annotations_list)}')

    model = models.resnext50_32x4d(pretrained=False)
    classes_count = len(predicted_classes)
    model.fc = nn.Linear(2048, classes_count)
    # model = nn.DataParallel(model, device_ids=device_ids, output_device=device)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint)
    model.to(running_device)
    model.eval()
    # criterion = nn.CrossEntropyLoss()

    classifier = torch.nn.Softmax()
    # создаём папки с классами
    if need_copy:
        current_dir = os.path.normpath(".")
        for class_name in predicted_classes:
            os.makedirs(fr'{os.path.join(current_dir, result_dir, class_name)}',
                        exist_ok=True)

    class_mapping = {idx: cls for idx, cls in enumerate(predicted_classes)}

    # classification results
    classified_list = []
    val_len = len(classify_loader)
    for idx, (imgs, filepaths) in tqdm(enumerate(classify_loader),
                                       total=val_len):
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
    pd.DataFrame(classified_list,
                 columns=result_columns).to_csv(result_path, sep=';')
    print("Сохранение завершено!")


if __name__ == '__main__':
    classify(
        dataset_path=TEST_DATASET_PATH,
        weights_path=MODEL_WEIGHTS_PATH,
        result_path=RESULT_FILE,
        result_dir=RESULT_DIR,
        batch_size=BATCH_SIZE,
        workers_num=NUM_WORKERS,
        predicted_classes=CLASSES,
        device=PREDICT_DEVICE,
        result_columns=RESULT_COLUMNS,
    )
