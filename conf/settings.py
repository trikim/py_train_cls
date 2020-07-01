

from datetime import datetime


#cv2
#TRAIN_MEAN = [0.43237713785804116, 0.49941626449353244, 0.48560741861744905]
#TRAIN_STD = [0.2665100547329813, 0.22770540015765814, 0.2321024260764962]

# TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
# TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]

#TEST_MEAN = [0.4311430419332438, 0.4998156522834164, 0.4862169586881995]
#TEST_STD = [0.26667253517177186, 0.22781080253662814, 0.23264268069040475]

#TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
#TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]

# distill mean and std (resnet152)
TRAIN_MEAN = [0.4850196078431373, 0.457956862745098, 0.4076039215686274]
TRAIN_STD = [0.229, 0.224, 0.225]

#DATA_PATH = '/media/lhy/DATA/train_data/02_face_attr'
DATA_SPLIT = ' '
#TRAIN_TXT = 'face_attr_train.txt'
#TEST_TXT = 'face_attr_train.txt'

#DATA_PATH = '/home/lhy/study/train_method/train_data/veh_color'
#DATA_SPLIT = ' , '
#TRAIN_TXT = 'vehrear_color_train.txt'
#TEST_TXT = 'vehrear_color_test.txt'

#DATA_PATH = '/home/study/train_method/train_data/veh_type_data'
#DATA_SPLIT = ' , '
#TRAIN_TXT = 'vehtype_label.txt'
#TEST_TXT = 'vehtype_test.txt'


DATA_PATH_IMAGENET = '/home/study/train_method/train_data/veh_type_data/veh_imagnet'
DATA_PATH_IMAGENET_VAL = '/home/study/train_method/train_data/veh_type_data/veh_imagnet_val'

#weights file directory
CHECKPOINT_PATH = 'checkpoints'
#tensorboard log file directory
LOG_DIR = 'runs'
TIME_NOW = datetime.now().isoformat()





