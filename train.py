import gc
import torchvision
import torch
import numpy as np
import json
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
import pandas as pd
import os
import time
import h5py
import argparse
import yaml
import psutil
import coco_dataset
import imat_dataset
import visualize
from datetime import datetime
from pycocotools.coco import COCO
import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
from torchvision.ops import MultiScaleRoIAlign
import engine, utils
import effdet
from effdet import BiFpn, DetBenchTrain, EfficientDet, load_pretrained, load_pretrained, HeadNet
import subprocess
import sys
from memory_profiler import profile
import functools
import socket
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

print = functools.partial(print, flush=True)

is_colab = False
try:
    from google.colab import drive
    print("Running on Google Colab")
    is_colab = True
except:
    print("Running on non Google Colab env")

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

# seeds
# following is needed for reproducibility
# refer to https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='Training Config')

# parsing boolean typed arguments
# refer to https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected but got [{}].'.format(v))

# training params
parser.add_argument('--model-name', type=str, default='tf_efficientdet_d0', metavar='MODEL_NAME',
                    help='The name of the model to use as found in EfficientDet model_config.py file (default=tf_efficientdet_d0)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight-decay', type=float, default=0.00005, metavar='WEIGHT_DECAY',
                    help='weight decay (default: 0.00005)')
parser.add_argument('--box-threshold', type=float, default=0.3, metavar='BOX_THRESHOLD',
                    help='score threshold - boxes with scores lower than specified will be ignored in training and evaluation (default: 0.3)')
parser.add_argument('--batch-size', type=int, default=12, metavar='BATCH_SIZE',
                    help='batch size (default: 12)')
parser.add_argument('--num-workers', type=int, default=4, metavar='NUM_WORKERS',
                    help='number of workers for the dataloader (default: 4)')
parser.add_argument('--num-epochs', type=int, default=150, metavar='NUM_EPOCHS',
                    help='number of epochs (default: 150)')
parser.add_argument('--model-file-suffix', type=str, default='effdet_d0', metavar='SUFFIX',  # TODO(ofekp): default should be empty string?
                    help='Suffix to identify the model file that is saved during training (default=effdet_h5py_rpn)')
parser.add_argument('--model-file-prefix', type=str, default='', metavar='PREFIX',  # TODO(ofekp): default should be empty string?
                    help='Prefix, may be folder, to load the model file that is saved during training (default=empty string)')
parser.add_argument('--add-user-name-to-model-file', type=str2bool, default=True, metavar='BOOL',
                    help='Will add the user name to the model file that is saved during training (default=True)')
parser.add_argument('--load-model', type=str2bool, default=False, metavar='BOOL',
                    help='Will load a model file (default=False)')
parser.add_argument('--train', type=str2bool, default=True, metavar='BOOL',
                    help='Will start training the model (default=True)')
parser.add_argument('--data-limit', type=int, default=12500, metavar='DATA_LIMIT',
                    help='Specify data limit, None to use all the data (default=12500)')
parser.add_argument('--target-dim', type=int, default=512, metavar='DIM',
                    help='Dimention of the images. It is vital that the image size will be devisiable by 2 at least 6 times (default=512)')
parser.add_argument('--h5py-dataset', type=str2bool, default=True, metavar='BOOL',
                    help='Use an H5PY dataset as created using h5py_dataset_writer.py (default=True)')
parser.add_argument('--coco-dataset', type=str2bool, default=True, metavar='BOOL',
                    help='Use COCO dataset (default=True)')
parser.add_argument('--freeze-batch-norm-weights', type=str2bool, default=True, metavar='BOOL',
                    help='Freeze batch normalization weights (default=True)')

# scheduler params
parser.add_argument('--sched-factor', type=float, default=0.5, metavar='FACTOR',
                    help='scheduler factor (default: 0.5)')
parser.add_argument('--sched-patience', type=int, default=1, metavar='PATIENCE',
                    help='scheduler patience (default: 1)')
parser.add_argument('--sched-verbose', type=str2bool, default=False, metavar='VERBOSE',
                    help='scheduler verbosity (default: False)')
parser.add_argument('--sched-threshold', type=float, default=0.0001, metavar='THRESHOLD',
                    help='scheduler threshold (default: 0.0001)')
parser.add_argument('--sched-min-lr', type=float, default=1e-8, metavar='MIN_LR',
                    help='scheduler min LR (default: 1e-8)')
parser.add_argument('--sched-eps', type=float, default=1e-08, metavar='EPS',
                    help='scheduler epsilon (default: 1e-08)')

# additional params
parser.add_argument('--gradient-accumulation-steps', type=int, default=2, metavar='NUM_EPOCHS',
                    help='number of epoch to accumulate gradients before applying back-prop (default: 2)')  # TODO(ofekp): change to 1?
parser.add_argument('--save-every', type=int, default=5, metavar='NUM_EPOCHS',
                    help='save the model every few epochs (default: 5)')
parser.add_argument('--eval-every', type=int, default=10, metavar='NUM_EPOCHS',
                    help='evaluate and print the evaluation to screen every few epochs (default: 10)')


def parse_args():
    # parse the args that are passed to this script
    args = parser.parse_args()

    # save the args as a text string so we can log them later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


current_time_millis = lambda: int(round(time.time() * 1000))


def print_bold(str):
    print("\033[1m" + str + "\033[0m")


def run_os(cmd_as_list):
    process = subprocess.Popen(cmd_as_list,
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.strip().decode('utf-8') if stdout is not None else stdout
    stderr = stderr.strip().decode('utf-8') if stderr is not None else stderr
    return stdout, stderr


def print_nvidia_smi(device):
    if device != 'cpu':
        stderr, _ = run_os(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv'])
        print(stderr)


def process_data(main_folder_path, data_limit):
    allowed_classes = None  # np.array([0,1,6,9,10,20,23,24,31,32,33])

    with open(main_folder_path + '/Data/label_descriptions.json', 'r') as file:
        label_desc = json.load(file)
    sample_sub_df = pd.read_csv(main_folder_path + '/Data/sample_submission.csv')
    data_df = pd.read_csv(main_folder_path + '/Data/train.csv')
    cut_data_df = data_df
    if allowed_classes is not None:
        print("Data is limited to segments for these class ids {} entires".format(allowed_classes))
        cut_data_df = cut_data_df[cut_data_df['ClassId'].isin(allowed_classes)]
        cut_data_df['ClassId'] = cut_data_df['ClassId'].apply(lambda x: np.where(allowed_classes == x)[0][0])

    image_ids = cut_data_df['ImageId'].unique()
    image_ids_cut = image_ids

    if data_limit is not None:
        print("Data is limited to [{}] images".format(data_limit))
        image_ids_cut = image_ids[:data_limit]

    image_train_count = int(len(image_ids_cut) * 0.8)
    image_ids_train = image_ids_cut[:image_train_count]
    image_ids_test = image_ids_cut[image_train_count:]
    assert len(image_ids_cut) == (len(image_ids_test) + len(image_ids_train))
    cut_data_df = cut_data_df[cut_data_df['ImageId'].isin(image_ids_cut)]
    train_df = cut_data_df[cut_data_df['ImageId'].isin(image_ids_train)]
    test_df = cut_data_df[cut_data_df['ImageId'].isin(image_ids_test)]
    assert len(cut_data_df) == (len(test_df) + len(train_df))

    print("Train data size [{}] test data size [{}] (counting in segments)".format(len(train_df), len(test_df)))
    print()
    num_classes = None
    if allowed_classes is None:
        num_classes = len(data_df['ClassId'].unique())
    else:
        num_classes = len(allowed_classes)
    num_attributes = len(label_desc['attributes'])
    print_bold("Classes")
    categories_df = pd.DataFrame(label_desc['categories'])
    if allowed_classes is not None:
        allowed_col = categories_df['id'].isin(allowed_classes)
        allowed_col = categories_df['id'].isin(allowed_classes)
        d = {True: 'V', False: ''}
        allowed_col = allowed_col.replace(d)
        categories_df['is_allowed'] = allowed_col
    attributes_df = pd.DataFrame(label_desc['attributes'])
    print(categories_df)
    print(f'Total # of classes: {num_classes}')
    print()
    print_bold("Attributes")
    print(attributes_df.head())
    print(f'Total # of attributes: {num_attributes}')
    print()
    train_df.head()
    return num_classes, train_df, test_df, categories_df


def process_data_coco(main_folder_path, data_limit):
    train_annot_path = main_folder_path + '/Data/annotations/instances_train2017.json'
    val_annot_path = main_folder_path + '/Data/annotations/instances_val2017.json'
    if socket.gethostname() == "deep3d":
        print("Running on Deep3D")
        train_annot_path = '/data/dataset/COCO_2017/annotations/instances_train2017.json'
        val_annot_path = '/data/dataset/COCO_2017/annotations/instances_val2017.json'
    coco_train = COCO(train_annot_path)
    coco_val = COCO(val_annot_path)
    # Load the categories in a variable
    cat_ids = coco_val.getCatIds()
    cats = coco_val.loadCats(cat_ids)
    num_classes = max([cat['id'] for cat in cats])
    print("Num of classes is [{}]".format(num_classes))
    # categories start from 0
    categories_df = pd.DataFrame([cat['name'] for cat in cats], columns=['name'], index=[cat['id'] - 1 for cat in cats])
    categories_df = categories_df.reindex(range(91), fill_value='none')  # in coco some categories are empty
    print(cats)
    print(categories_df)
    return num_classes, coco_train, coco_val, categories_df


# h5py
class DatasetH5Reader(torch.utils.data.Dataset):
    def __init__(self, in_file):
        super(DatasetH5Reader, self).__init__()
        self.in_file = in_file
        
#         assert self.h5py_file.swmr_mod
#         self.n_images, self.nx, self.ny = self.file['images'].shape

    def __getitem__(self, index):
        h5py_file = h5py.File(self.in_file, "r", swmr=True)  # swmr=True allows concurrent reads
        image = h5py_file['images'][index]
        labels = h5py_file['labels'][index]
        masks_fixed_size = h5py_file['masks'][index]
        boxes_fixed_size = h5py_file['boxes'][index]
        return image, labels, masks_fixed_size, boxes_fixed_size

    def __len__(self):
        h5py_file = h5py.File(self.in_file, "r", swmr=True)  # swmr=True allows concurrent reads
        return h5py_file['images'].shape[0]


def set_bn_eval(m):
    classname = m.__class__.__name__
    if "BatchNorm2d" in classname:
        m.affine = False
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m.eval()


def freeze_bn(model):
    model.apply(set_bn_eval)
    

def get_model_instance_segmentation(num_classes, rank):
    """
    This is the conventional model which is based on Faster R-CNN
    Note that to use this model you must install regular pytorch package (instead of from ofekp branch)
    and use '--model-name faster' in the arguments
    The correct way to install torchvision will be:
        pip uninstall torchvision
        pip install torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    To restore back to EfficientDet use:
        pip uninstall torchvision
        pip install git+https://github.com/ofekp/vision.git

    UPDATE 2021, to use this execute:
        pip3 uninstall torch
        pip3 uninstall torchvision
        pip3 install torch
        pip3 install torchvision
        this will give torch 1.9.0, and vision of 0.10.0 which are compatible based on the table in
        https://github.com/pytorch/vision
    """
    print("Using Faster-RCNN detection model")
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    if torch.cuda.device_count() > 1:
        print("Found rank [{}]".format(rank))
        print_nvidia_smi(rank)
        model.to(rank)
        model = DDP(model, device_ids=[rank])

    return model


class EfficientDetBB(nn.Module):

    def __init__(self, config, class_net, box_net):
        super(EfficientDetBB, self).__init__()
        self.class_net = class_net
        self.box_net = box_net

    def forward(self, x):
        '''
        Originally EfficientDet also conatined the backbone and then fpn
        but for the purpose of our network this had to be modified
        '''
        x_class = self.class_net(x)
        x_box = self.box_net(x)
        return x_class, x_box

    
class BackboneWithCustomFPN(nn.Module):
    def __init__(self, config, backbone, fpn, out_channels, alternate_init=False):
        super(BackboneWithCustomFPN, self).__init__()
        self.body = backbone
        self.fpn = fpn
        self.out_channels = out_channels
        
        for n, m in self.named_modules():
            if 'body' not in n and 'backbone' not in n:  # avoid changing the weights of the backbone which is pretrained
                if alternate_init:
                    effdet._init_weight_alt(m, n)
                else:
                    effdet._init_weight(m, n)

    def forward(self, x):
        '''
        Args:
            x - in BCHW format, e.g. x.shape = torch.Size([2, 3, 512, 512])
        '''
        x = self.body(x)  # len(x) = 3
        x = self.fpn(x)
        # at this point x is an OrderedDict of features
        return x


def get_model_instance_segmentation_efficientnet(model_name, num_classes, target_dim, rank, freeze_batch_norm=False):
    print("Using EffDet detection model")
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)
    # ofekp: note that roi_pooler is passed to box_roi_pooler in the MaskRCNN network
    # and is not being used in roi_heads.py
    
    mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=14,
                sampling_ratio=2)
    
    config = effdet.get_efficientdet_config(model_name)
    efficientDetModelTemp = EfficientDet(config, pretrained_backbone=False)
    load_pretrained(efficientDetModelTemp, config.url)
    config.num_classes = num_classes
    config.image_size = target_dim

    out_channels = config.fpn_channels  # This is since the config of 'tf_efficientdet_d5' creates fpn outputs with num of channels = 288
    backbone_fpn = BackboneWithCustomFPN(config, efficientDetModelTemp.backbone, efficientDetModelTemp.fpn, out_channels)  # TODO(ofekp): pretrained! # from the repo trainable_layers=trainable_backbone_layers=3
    model = MaskRCNN(backbone_fpn,
                 min_size=target_dim,
                 max_size=target_dim,
                 num_classes=num_classes,
                 mask_roi_pool=mask_roi_pool,
#                  rpn_anchor_generator=anchor_generator,
                 box_roi_pool=roi_pooler)
    
    # for training with different number of classes (default is 90) we need to add this line
    # TODO(ofekp): we might want to init weights of the new HeadNet
    class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    efficientDetModel = EfficientDetBB(config, class_net, efficientDetModelTemp.box_net)
    model.roi_heads.box_predictor = DetBenchTrain(efficientDetModel, config)

    if freeze_batch_norm:
        # we only freeze BN layers in backbone and the BiFPN
        print("Freezing batch normalization weights")
        freeze_bn(model.backbone)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    return model


class Trainer:
    
    def __init__(self, main_folder_path, model, train_df, test_df, data_limit, num_classes, target_dim, categories_df, device, is_colab, rank, world_size, config):
        self.main_folder_path = main_folder_path
        self.model = model
        self.train_df = train_df
        self.rank = rank
        self.world_size = world_size
        self.test_df = test_df
        self.device = device
        self.config = config
        self.num_classes = num_classes
        self.target_dim = target_dim
        self.is_colab = is_colab
        self.data_limit = data_limit
        if "faster" in self.config.model_name:
            # special case of training the conventional model based on Faster R-CNN
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = self.config.optimizer_class(params, **self.config.optimizer_config)
        else:
            self.optimizer = self.config.optimizer_class(self.model.parameters(), **self.config.optimizer_config)
        self.scheduler = self.config.scheduler_class(self.optimizer, **self.config.scheduler_config)
        self.model_file_path = self.get_model_file_path(is_colab, prefix=config.model_file_prefix, suffix=config.model_file_suffix)
        self.log_file_path = self.get_log_file_path(is_colab, suffix=config.model_file_suffix)
        self.epoch = 0
        self.visualize = visualize.Visualize(self.main_folder_path, categories_df, self.target_dim, dest_folder='Images')
        self.log("Memory usage before initializing the datasets [{}]".format(get_memory_usage()))
        # use our dataset and defined transformations
        if self.config.coco_dataset:
            if self.config.h5py_dataset:
                h5_reader = imat_dataset.DatasetH5Reader("/data/dataset/COCO_2017_H5PY/coco_" + str(self.target_dim) + ".hdf5")
                self.dataset = imat_dataset.IMATDatasetH5PY(h5_reader, self.num_classes, self.target_dim, self.config.model_name, T.get_transform(train=True))
                h5_reader_test = imat_dataset.DatasetH5Reader("/data/dataset/COCO_2017_H5PY/coco_test_" + str(self.target_dim) + ".hdf5")
                self.dataset_test = imat_dataset.IMATDatasetH5PY(h5_reader_test, self.num_classes, self.target_dim, self.config.model_name, T.get_transform(train=False))
            else:
                self.dataset = coco_dataset.COCODataset(self.train_df, True, self.config.model_name, self.main_folder_path, self.target_dim, self.num_classes,
                                                        T.get_transform(train=True), False)
                self.dataset_test = coco_dataset.COCODataset(self.test_df, False, self.config.model_name, self.main_folder_path, self.target_dim, self.num_classes,
                                                            T.get_transform(train=False), False)
        else:
            if self.config.h5py_dataset:
                h5_reader = imat_dataset.DatasetH5Reader("../imaterialist_" + str(self.target_dim) + ".hdf5")
                self.dataset = imat_dataset.IMATDatasetH5PY(h5_reader, self.num_classes, self.target_dim, self.config.model_name, T.get_transform(train=True))
                h5_reader_test = imat_dataset.DatasetH5Reader("../imaterialist_test_" + str(self.target_dim) + ".hdf5")
                self.dataset_test = imat_dataset.IMATDatasetH5PY(h5_reader_test, self.num_classes, self.target_dim, self.config.model_name, T.get_transform(train=False))
            else:
                self.dataset = imat_dataset.IMATDataset(self.main_folder_path, self.train_df, self.num_classes, self.target_dim, self.config.model_name, False, T.get_transform(train=True))
                self.dataset_test = imat_dataset.IMATDataset(self.main_folder_path, self.test_df, self.num_classes, self.target_dim, self.config.model_name, False, T.get_transform(train=False))
        self.log("Memory usage after initializing the datasets [{}]".format(get_memory_usage()))
        # TODO(ofekp): do we need this?
        # split the dataset in train and test set
        # indices = torch.randperm(len(dataset)).tolist()
        # dataset = torch.utils.data.Subset(dataset, indices[:-50])
        # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        self.log('Trainer initialized. Device is [{}]'.format(self.device))

    def get_model_identifier(self):
        return 'dim_' + str(self.target_dim) + '_images_' + str(self.data_limit) + '_classes_' + str(self.num_classes)

    def get_model_file_path(self, is_colab, prefix=None, suffix=None):
        model_file_path = self.get_model_identifier()
        if prefix:
            model_file_path = prefix + ('' if prefix[-1] == '/' else '_') + model_file_path
        if suffix:
            model_file_path = model_file_path + '_' + suffix
            
        model_file_path = 'Model/' + model_file_path + '.model'

        if is_colab:
            model_file_path = self.main_folder_path + 'code_ofek/' + model_file_path
        else:
            model_file_path = self.main_folder_path + 'Code/' + model_file_path
        
        return model_file_path

    def get_log_file_path(self, is_colab, prefix=None, suffix=None):
        log_file_path = self.get_model_identifier()
        if prefix:
            log_file_path = prefix + ('' if prefix[-1] == '/' else '_') + log_file_path
        if suffix:
            log_file_path = log_file_path + '_' + suffix
            
        log_file_path = 'Log/' + log_file_path + '.log'

        if is_colab:
            log_file_path = self.main_folder_path + 'code_ofek/' + log_file_path
        else:
            log_file_path = self.main_folder_path + 'Code/' + log_file_path
        
        return log_file_path

    def load_model(self, device):
        if not os.path.isfile(self.model_file_path):
            self.log("Cannot load model file [{}] since it does not exist".format(self.model_file_path))
            return False
        checkpoint = torch.load(self.model_file_path)  # map_location=device
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # model must be moved to device before we init the optimizer otherwise loading a model and training
        # again will procduce the "both cpu and cuda" error, refer to the solution in this thread:
        # https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783
        # I also added the solution to this issue https://github.com/pytorch/pytorch/issues/34470
        self.model.to(device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # TODO(ofekp): uncomment
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # TODO(ofekp): uncomment
#         self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        self.log("Loaded model file [{}] trained epochs [{}]".format(self.model_file_path, checkpoint['epoch']))
        return True
        
    def save_model(self):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
#             'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, self.model_file_path)        
        self.log('Saved model to [{}]'.format(self.model_file_path))
        print_nvidia_smi(self.device)
        self.dataset.show_stats()

    def eval_model(self, data_loader_test):
        print("Memory usage before starting eval [{}]".format(psutil.virtual_memory().percent), flush=True)
        self.model.eval()
        with torch.no_grad():
            img_idx = 0
            self.visualize.show_prediction_on_img(self.model, self.dataset_test, self.test_df, img_idx, self.is_colab, show_ground_truth=False, box_threshold=self.config.box_threshold, split_segments=False)
            # evaluate on the test dataset
            if "faster" in self.config.model_name:
                # special case of training the conventional model based on Faster R-CNN
                engine.evaluate(self.model, data_loader_test, device=self.device, box_threshold=None)
            else:
                engine.evaluate(self.model, data_loader_test, device=self.device)
            print("Memory usage after eval [{}]".format(psutil.virtual_memory().percent), flush=True)
                
    def log(self, message):
        if self.config.verbose:
            print(message, flush=True)
        with open(self.log_file_path, 'a+') as logger:
            logger.write(f'{message}\n')

    def train(self):
        # define training and validation data loaders
        train_sampler = None
        test_sampler = None
        train_shuffle = True
        if torch.cuda.device_count() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )

            test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset_test,
                num_replicas=self.world_size,
                rank=self.rank
            )

            train_shuffle = False

        data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.config.batch_size, shuffle=train_shuffle, num_workers=self.config.num_workers,
            collate_fn=utils.collate_fn, pin_memory=True, sampler=train_sampler)

        data_loader_test = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers,
            collate_fn=utils.collate_fn, pin_memory=True, sampler=test_sampler)

        # self.eval_model(data_loader_test)  # TODO: remove this line

        for _ in range(self.config.num_epochs):
            # train one epoch
            metric_logger = engine.train_one_epoch(
                self.model,
                self.optimizer,
                data_loader,
                self.device,
                self.epoch,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                print_freq=100,
                box_threshold=self.config.box_threshold)

            # update the learning rate
            if "_d0" in self.config.model_name:
                print("Updating StepLR")
                self.scheduler.step()
            else:
                print("Updating ReduceLROnPlateau")
                self.scheduler.step(metric_logger.__getattr__('loss').avg)
            torch.cuda.empty_cache()  # ofekp: attempting to avoid GPU memory usage increase

            if (self.epoch) % self.config.save_every == 0:
                self.save_model()
            
            if (self.epoch) % self.config.eval_every == 0:
                self.eval_model(data_loader_test)
            
            self.log("Epoch [{}/{}]".format(self.epoch + 1, self.config.num_epochs))
            self.log("Memory usage after epoch [{}]".format(get_memory_usage()))
            self.epoch += 1

        self.log("Saving model one last time")
        self.save_model()
        self.eval_model(data_loader_test)
        self.log("That's it!")

        
class TrainConfig:
    def __init__(self, args):
        if args.add_user_name_to_model_file:
            self.model_file_suffix = os.getlogin() + "_" + args.model_file_suffix
        else:
            self.model_file_suffix = args.model_file_suffix
        self.model_file_prefix = args.model_file_prefix
        self.h5py_dataset = args.h5py_dataset
        self.coco_dataset = args.coco_dataset
        self.verbose = True
        self.save_every = args.save_every
        self.eval_every = args.eval_every
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_epochs = args.num_epochs
        self.model_name = args.model_name
        self.box_threshold = args.box_threshold

        # optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.00005)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        if "faster" in self.model_name:
            # special case of training the conventional model based on Faster R-CNN
            self.optimizer_class = torch.optim.SGD
            self.optimizer_config = dict(
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
        else:
            self.optimizer_class = torch.optim.AdamW
            self.optimizer_config = dict(
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        
        if "_d0" in self.model_name:
            print("Using StepLR")
            self.scheduler_class = torch.optim.lr_scheduler.StepLR
            self.scheduler_config = dict(
                step_size=10,
                gamma=0.2
            )
        else:
            print("Using ReduceLROnPlateau")
            self.scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
            self.scheduler_config = dict(
                mode='min',
                factor=args.sched_factor,
                patience=args.sched_patience,
                verbose=False, 
                threshold=args.sched_threshold,
                threshold_mode='abs',
                cooldown=0, 
                min_lr=args.sched_min_lr,
                eps=args.sched_eps
        )


def get_memory_usage():
    return psutil.virtual_memory().percent


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if torch.cuda.device_count() > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    args, args_text = parse_args()

    print("Memory usage on start [{}]".format(get_memory_usage()))

    main_folder_path = "../"
        
    if "faster" in args.model_name:
        # special case of training the conventional model based on Faster R-CNN
        args.box_threshold = None

    if not os.path.exists("Args"):
        os.mkdir("Args")
    with open("Args/args_text.yml", 'w') as args_file:
        args_file.write(args_text)

    # create folders if needed
    needed_folders = ["./Model/", "./Log/"]
    for needed_folder in needed_folders:
        if not os.path.exists(needed_folder):
            os.mkdir(needed_folder)

    # prepare a log file
    now = datetime.now() # current date and time
    date_str = now.strftime("%Y%m%d%H%M")
    if torch.cuda.device_count() > 1:
        log_file_path = "./Log/" + date_str + "_rank_" + str(rank) + ".log"
    else:
        log_file_path = "./Log/" + date_str + ".log"
    log_file = open(log_file_path, "a")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    isTPU = False
    forceCPU = False
    if isTPU:
        device = xm.xla_device()
    elif forceCPU:
        device = 'cpu'
    elif torch.cuda.device_count() > 1:
        device = rank   
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
    print("Device type [{}]".format(device))
    if device != 'cpu':
        print("Device description [{}]".format(torch.cuda.get_device_name(rank)))

    print("Memory usage before data processing [{}]".format(get_memory_usage()))

    if args.coco_dataset:
        # currently data limit has no effect here on coco dataset
        num_classes, train_df, test_df, categories_df = process_data_coco(main_folder_path, args.data_limit)
    else:
        num_classes, train_df, test_df, categories_df = process_data(main_folder_path, args.data_limit)
    print("Setting target_dim to [{}]".format(args.target_dim))

    print("Memory usage after data processing [{}]".format(get_memory_usage()))

    if "faster" in args.model_name:
        # special case of training the conventional model based on Faster R-CNN
        model = get_model_instance_segmentation(num_classes, device)
    else:
        model = get_model_instance_segmentation_efficientnet(args.model_name, num_classes, args.target_dim, device, freeze_batch_norm=args.freeze_batch_norm_weights)

    print("Memory usage after initializing the model [{}]".format(get_memory_usage()))

    # get the model using our helper function
    train_config = TrainConfig(args)
    trainer = Trainer(main_folder_path, model, train_df, test_df, args.data_limit, num_classes, args.target_dim, categories_df, device, is_colab, rank, world_size, config=train_config)

    # load a saved model
    if args.load_model:
        if not trainer.load_model(device):
            exit(1)

    if args.train:
        if not torch.cuda.device_count() > 1:
            print_nvidia_smi(device)
            model.to(device)
        print_nvidia_smi(device)
        trainer.train()

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_file.close()


def main():
    if torch.cuda.device_count() > 1:
        try:
            print("This should only be printed once!")
            world_size = 2
            mp.spawn(run,
                args=(world_size,),
                nprocs=world_size,
                join=True)
        finally:
            dist.destroy_process_group()
    else:
        run(None, None)


if __name__ == '__main__':
    main()
