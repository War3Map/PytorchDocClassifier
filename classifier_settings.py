import os


# домашняя директория
HOME_DIR = r'/home/.../'

# путь к обученной модели(куда сохранять, откуда загружать)
MODEL_WEIGHTS_PATH = '/home/.../weight/weight_best.pth'
# путь к датасету
DATASET_PATH = r'/home/.../.../'
# путь к тренировочному набору
TRAIN_PATH = r'/home/.../.../...'
# путь к тестовому набору
TEST_DATASET_PATH = r'/home/.../.../...'

# Настройки для запуска режима обучения и тренировки
BATCH_SIZE = 4
NUM_WORKERS = 8
IMAGE_SIZE = 1024
N_EPOCHS = 50

# Размер картинки для тренировки
IMAGE_SIZE = 1024

# режим для запуска run_classifier
MODE = 'test'
# список классов
CLASSES = ['печатный', 'рукописный']

# сводный файл с результатами
RESULT_FILE = r"/home/.../result.csv"

# НЕ ИСПОЛЬЗУЕТСЯ каталог куда перемещаются файлы после классификации
RESULT_DIR = r"/home/.../results"

# колонки для сводного файла с результатами
RESULT_COLUMNS = ['Path', 'Predict']

# путь к датасету для тренировки
DATASET_CSV_PATH = "/home/.../train.csv"
# разделитель внутри
CSV_DELIMITER = ','

# Путь к датасету для разметки
DS_PATH = r"...\path"

# Какое устройство будет использоваться для предсказания 'cuda:0' или 'cpu'
PREDICT_DEVICE = 'cuda:0'
# Путь к исходной модели, например resnext
# В случае когда есть доступ в интернет не треуется
# os.environ['TORCH_HOME'] = "/home/.../"
