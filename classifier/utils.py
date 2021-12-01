import pandas as pd
from albumentations import Compose, Resize, HorizontalFlip, Normalize, RandomBrightnessContrast, RandomGamma, \
    GaussNoise, ShiftScaleRotate, ImageCompression, CoarseDropout
from albumentations.pytorch import ToTensorV2
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


def get_train_augmentations(image_size):
    return Compose([
        Resize(image_size, image_size),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.4, brightness_limit=0.25, contrast_limit=0.3),
        RandomGamma(p=0.4),
        CoarseDropout(p=0.1, max_holes=8, max_height=8, max_width=8),
        GaussNoise(p=0.1, var_limit=(5.0, 50.0)),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.8),
        ImageCompression(quality_lower=80, quality_upper=100, p=0.4),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_augmentations(image_size):
    return Compose([
        Resize(image_size, image_size),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.8),
        ImageCompression(quality_lower=80, quality_upper=100, p=0.4),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.4, brightness_limit=0.25, contrast_limit=0.3),
        RandomGamma(p=0.4),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])


def preprocess_data(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    fold_index = 0
    for t1, t2 in skf.split(df['path'], df['label']):
        df.at[t2, 'fold'] = fold_index
        fold_index += 1
    train_df = df[df['fold'] != 3]
    val_df = df[df['fold'] == 3]
    le = preprocessing.LabelEncoder()
    le = le.fit(train_df['label'].values)
    train_labels = le.transform(train_df['label'].values)
    val_labels = le.transform(val_df['label'].values)
    # train_df.reset_index(inplace=True)
    # val_df.reset_index(inplace=True)
    return train_df, val_df, train_labels, val_labels
