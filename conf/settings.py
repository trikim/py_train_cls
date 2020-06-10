from datetime import datetime

TRAIN_MEAN = [0.485, 0.456, 0.406]
TRAIN_STD = [0.229, 0.224, 0.225]

DATA_SPLIT = ' '

DATA_PATH_IMAGENET = '/home/data/14'
DATA_PATH_IMAGENET_VAL = '/home/data/14'

#weights file directory
CHECKPOINT_PATH = '/project/train/models'
#tensorboard log file directory
LOG_DIR = 'runs'
TIME_NOW = datetime.now().isoformat()
IMAGE_SIZE = 96
