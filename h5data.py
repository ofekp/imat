import multiprocessing
import h5py

import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import json
import math
import re
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
from sklearn import svm
from keras.datasets import fashion_mnist
import pandas as pd
import joblib
import os
import random
import time
import multiprocessing
from multiprocessing import Process, Pool, Queue, Lock, Value
import itertools
import importlib
from PIL import Image
import h5py

# imports for segmentation
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

# imports external packages (from folder)
import bounding_box as bbx
import worker
import utils
import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops
import pycocotools
import coco_utils, coco_eval, engine, utils
from timm.models.layers import get_act_layer
from timm import create_model
import effdet
from effdet import BiFpn, DetBenchTrain, EfficientDet, load_pretrained, load_pretrained, HeadNet

# seeds
# following is needed for reproducibility
# refer to https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

# 2

main_folder_path = '../'
is_colab = False

current_time_millis = lambda: int(round(time.time() * 1000))

def print_bold(str):
    print("\033[1m" + str + "\033[0m")

    
def get_image_path(image_id):
    if is_colab:
        # images are split into shards (folders) to avoid too many files in one folder
        shard_folder = image_id[0:2]
        return main_folder_path + f'/Data/shard/train/{shard_folder}/{image_id}.jpg'
    else:
        # non-sharded images
        return main_folder_path + f'/Data/train/{image_id}.jpg'
    
    
def get_model_identifier():
    return 'dim_' + str(target_dim) + '_images_' + str(limit_data) + '_classes_' + str(num_classes)


def get_model_file_path(prefix=None, suffix=None):
    model_file_path = get_model_identifier()
    if prefix:
        model_file_path = prefix + '_' + model_file_path
    if suffix:
        model_file_path = model_file_path + '_' + suffix
        
    model_file_path = 'Model/' + model_file_path + '.model'

    if is_colab:
        model_file_path = main_folder_path + 'code_ofek/' + model_file_path
    else:
        model_file_path = main_folder_path + 'Code/' + model_file_path
    
    return model_file_path


def get_log_file_path(prefix=None, suffix=None):
    log_file_path = get_model_identifier()
    if prefix:
        log_file_path = prefix + '_' + log_file_path
    if suffix:
        log_file_path = log_file_path + '_' + suffix
        
    log_file_path = 'Log/' + log_file_path + '.log'

    if is_colab:
        log_file_path = main_folder_path + 'code_ofek/' + log_file_path
    else:
        log_file_path = main_folder_path + 'Code/' + log_file_path
    
    return log_file_path


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device type [{}]".format(device))
if device == 'cuda:0':
    print("Device description [{}]".format(torch.cuda.get_device_name(0)))
    
# 3

# TODO(ofekp): these should be None to avoid limiting
limit_data = 10000
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

if limit_data is not None:
    print("Data is limited to [{}] images".format(limit_data))
    image_ids_cut = image_ids[:limit_data]

image_train_count = int(len(image_ids_cut) * 0.8)
image_ids_train = image_ids_cut[:image_train_count]
image_ids_test = image_ids_cut[image_train_count:]
assert len(image_ids_cut) == (len(image_ids_test) + len(image_ids_train))
cut_data_df = cut_data_df[cut_data_df['ImageId'].isin(image_ids_cut)]
train_df = cut_data_df[cut_data_df['ImageId'].isin(image_ids_train)]
test_df = cut_data_df[cut_data_df['ImageId'].isin(image_ids_test)]
assert len(cut_data_df) == (len(test_df) + len(train_df))

print("Train data size [{}] test data size [{}]".format(len(train_df), len(test_df)))
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

# 4

target_dim = (2 ** 6) * 8  # it is vital that the image size will be devisiable by 2 at least 5 times
font = bbx.get_font_with_size(10)
print("Setting target_dim to [{}]".format(target_dim))


def get_transform(train):
    transforms = []
#     transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomGreyscale(0.1))
    return T.Compose(transforms)


class IMATDataset(BaseDataset):
    def __init__(self, data_df, transforms=None):
        self.transforms = transforms
        self.data_df = data_df
        self.image_ids = data_df['ImageId'].unique()
        self.skipped_images = []
        # TODO: indices = torch.randperm(len(dataset)).tolist()
        
    def show_stats(self):
        images_processed = self.images_processed.value
        total_process_time = self.total_process_time.value
        avg_time_per_image = 0 if images_processed == 0 else total_process_time / images_processed
        avg_image_load_time = 0 if images_processed == 0 else self.total_image_load_time.value / images_processed
        avg_transform_time = 0 if images_processed == 0 else self.total_transform_time.value / images_processed
        avg_mask_time = 0 if images_processed == 0 else self.total_mask_time.value / images_processed
        avg_box_time = 0 if images_processed == 0 else self.total_box_time.value / images_processed
        print("Processed [{}] images in [{}] seconds."
              "Avg of [{}] per image."
              "Avg image load time [{}]"
              "Avg transform time [{}]"
              "Avg mask time [{}]"
              "Avg box time [{}]"
              .format(
            images_processed,
            total_process_time,
            avg_time_per_image,
            avg_image_load_time,
            avg_transform_time,
            avg_mask_time,
            avg_box_time))
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        vis_df = self.data_df[self.data_df['ImageId'] == image_id]
        vis_df = vis_df.reset_index(drop=True)
        labels = worker.get_labels(vis_df)
        mask_start_ts = time.time()
        try:
            masks = worker.get_masks(vis_df, target_dim=target_dim)
            for mask in masks:
                assert not torch.any(torch.isnan(mask))
                assert torch.where(mask > 0)[0].shape[0] == torch.sum(mask)  # check only ones and zeros
        except Exception as e:
            self.skipped_images.append(image_id)
            print("ERROR: Skipped image with id [{}] due to a mask exception [{}]".format(image_id, e))
            return
        
        box_start_ts = time.time()
        boxes = worker.get_bounding_boxes(vis_df, masks)
        try:
            for box in boxes:
                assert not torch.any(torch.isnan(box))
        except Exception as e:
            self.skipped_images.append(image_id)
            print("ERROR: Skipped image with id [{}] due to a BB exception [{}]".format(image_id, e))
            return
        
        num_objs = len(labels)

        image_id_idx = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        labels, masks, boxes = worker.remove_empty_masks(labels, masks, boxes)

        target = {}
        target["labels"] = torch.add(labels, 1)  # refer to fast_collate, this is needed for efficient det
        assert torch.min(target["labels"]) >= 1
        assert torch.max(target["labels"]) <= num_classes
        target["masks"] = masks
        target["boxes"] = boxes
        target["image_id"] = image_id_idx
        target["area"] = area
        target["iscrowd"] = iscrowd
#         target["image_id"] = torch.tensor(image_id_idx)
#         target["area"] = torch.tensor(area)
#         target["iscrowd"] = torch.tensor(iscrowd)

        image_load_start_ts = time.time()
        image_orig = Image.open(get_image_path(image_id)).convert("RGB")
        # image = mpimg.imread(get_image_path(image_id))
        image = worker.rescale(image_orig, target_dim=target_dim)
        
        # TODO(ofekp): make sure that this makes sense!
        # TODO(ofekp): check what happens here when the image is < target_dim. What will worker.py scale method do to the image in this case?
        target["img_size"] = image_orig.size[-2:] if target_dim is None else (target_dim, target_dim)
        image_orig_max_dim = max(target["img_size"])
        img_scale = target_dim / image_orig_max_dim
        target["img_scale"] = 1. / img_scale  # back to original size
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        assert image.shape[0] <= target_dim and image.shape[1] <= target_dim and image.shape[2] <= target_dim
        return image, target

    def __len__(self):
        return len(self.image_ids)


class DatasetH5Writer(torch.utils.data.Dataset):
    def __init__(self, dataset, out_file, chunk_size):
        super(DatasetH5Writer, self).__init__()
#         print(data_df['ImageId'])
#         self.image_ids = data_df['ImageId'].unique()
#         print(self.image_ids)
#         self.lock = Lock()
        self.dataset = dataset
        self.dataset_len = self.dataset.__len__()
        self.chunk_size = chunk_size
        self.file_name = out_file
        self.cpu_count = multiprocessing.cpu_count()
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        self.h5py_file = h5py.File(self.file_name, "a")
        self.image_ids_data_set = self.h5py_file.create_dataset("image_ids", shape=(0,), dtype=np.uint64, maxshape=(None,), chunks=(self.chunk_size,))
        self.images_data_set = self.h5py_file.create_dataset("images", shape=(0, 3, target_dim, target_dim), dtype=np.float64, maxshape=(None, 3, target_dim, target_dim), chunks=(self.chunk_size, 3, target_dim, target_dim))
        dt = h5py.vlen_dtype(np.dtype('int64'))
        self.labels_data_set = self.h5py_file.create_dataset("labels", shape=(0,), maxshape=(None,), dtype=dt, chunks=(self.chunk_size,))
        self.masks_data_set = self.h5py_file.create_dataset("masks", shape=(0,75,target_dim,target_dim), maxshape=(None,75,512,512), dtype=np.uint8, chunks=(self.chunk_size,75,target_dim,target_dim))
        self.boxes_data_set = self.h5py_file.create_dataset("boxes", shape=(0,75,4), maxshape=(None,75,4), dtype=np.float64, chunks=(self.chunk_size,75,4))
    
    def append_to_h5py(self, result):
        chunk_size, images_np, image_ids_np, labels_numpy_list, masks_numpy_fixed_size, boxes_numpy_fixed_size = result
        assert images_np.shape[0] == chunk_size
        assert len(masks_numpy_fixed_size) == chunk_size
        assert len(image_ids_np) == chunk_size
#         self.lock.acquire()
#         try:
        self.h5py_file = h5py.File(self.file_name, "a")

        curr_len = self.images_data_set.shape[0]
        self.images_data_set.resize(curr_len + chunk_size, axis=0)
        self.images_data_set[-chunk_size:] = images_np

        self.image_ids_data_set.resize(curr_len + chunk_size, axis=0)
        self.image_ids_data_set[-chunk_size:] = image_ids_np

        self.labels_data_set.resize(curr_len + chunk_size, axis=0)
        for i, labels_numpy in enumerate(labels_numpy_list):
            self.labels_data_set[curr_len + i] = labels_numpy

        self.masks_data_set.resize(curr_len + chunk_size, axis=0)
        self.masks_data_set[-chunk_size:] = masks_numpy_fixed_size

        self.boxes_data_set.resize(curr_len + chunk_size, axis=0)
        self.boxes_data_set[-chunk_size:] = boxes_numpy_fixed_size
        print("Dataset [{}] size is [{}]".format(self.file_name, self.images_data_set.shape[0]))
#         finally:
#             self.lock.release()

    @staticmethod
    def tensor_list_to_numpy(tensor_list):
        num_of_tensors = len(tensor_list)
        if num_of_tensors == 0:
            return np.array([])
        tensor_shape = tensor_list[0].shape
        for i, _ in enumerate(tensor_list):
            tensor_list[i] = tensor_list[i].unsqueeze(0)

        res_tensor = torch.Tensor(num_of_tensors, *tensor_shape)
        torch.cat(tensor_list, out=res_tensor)
        return res_tensor.numpy()
    
    
    @staticmethod
    def process_chunk(dataset, start_idx, chunk_size):
        dataset_len = dataset.__len__()
        curr_chunk_size = 0
        images = []
        image_ids = []
        labels_numpy_list = []
        masks_numpy_list = []
        boxes_numpy_list = []
        for _ in range(chunk_size):
            idx = start_idx + curr_chunk_size
            image, target = dataset.__getitem__(idx)
            if len(target["labels"]) == 0:
                print("Skipping image on index [{}] since it has empty labels".format(idx))
                continue
            images.append(image)
            image_ids.append(idx)
            labels_numpy_list.append(target["labels"].numpy())
            masks_numpy_list.append(target["masks"].numpy())
            boxes_numpy_list.append(target["boxes"].numpy())
            curr_chunk_size += 1
            if start_idx + curr_chunk_size == dataset_len:
                break
        images_numpy = DatasetH5Writer.tensor_list_to_numpy(images)
        image_ids_numpy = np.array(image_ids)
        masks_numpy_fixed_size = np.zeros((curr_chunk_size, 75, target_dim, target_dim), dtype=np.uint8)
        for i, masks_numpy in enumerate(masks_numpy_list):
            for j, mask in enumerate(masks_numpy):
                masks_numpy_fixed_size[i, j, :, :] = mask
        boxes_numpy_fixed_size = np.zeros((curr_chunk_size, 75, 4), dtype=np.float64)
        for i, boxes_numpy in enumerate(boxes_numpy_list):
            for j, box in enumerate(boxes_numpy):
                boxes_numpy_fixed_size[i, j, :] = box
        return (curr_chunk_size, images_numpy, image_ids_numpy, labels_numpy_list, masks_numpy_fixed_size, boxes_numpy_fixed_size)

    def process(self):
        print("CPU count is [{}]".format(self.cpu_count))
        print("Started writing [{}]...".format(self.file_name))
        pool = multiprocessing.Pool(self.cpu_count - 1)
        queue = multiprocessing.Queue()
        idx = 0
#         results = []
        count_chunks = 0
        while idx < self.dataset_len:
            pool.apply_async(DatasetH5Writer.process_chunk, (self.dataset, idx, self.chunk_size), callback=queue.put)
            idx += self.chunk_size
            count_chunks += 1
#             results.append(res)
# #         rrrr = [t.get() for t in results]
#         for res in results:
#             res.wait()
#             print(res)
#             print(res.get())
        pool.close()
    
        count_chunks_done = 0
        while True:
            if not queue.empty():
                print("Approx. queue size [{}]".format(queue.qsize()))
                count_chunks_done += 1
                self.append_to_h5py(queue.get())
                print("Processed chunks [{}/{}]".format(count_chunks_done, count_chunks))
                if count_chunks_done == count_chunks:
                    break
                    
        pool.join()
        print("File [{}] completed.".format(self.file_name))
            
    def close(self):  
        self.h5py_file.close()


dataset_test = IMATDataset(test_df, get_transform(train=False))
h5_test_writer = DatasetH5Writer(dataset_test, "../imaterialist_test_" + str(target_dim) + ".hdf5", chunk_size=20)
h5_test_writer.process()
h5_test_writer.close()
                
dataset = IMATDataset(train_df, get_transform(train=False))
h5_writer = DatasetH5Writer(dataset, "../imaterialist_" + str(target_dim) + ".hdf5", chunk_size=20)
h5_writer.process()
h5_writer.close()

print("All done.")
