import multiprocessing
import h5py

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
import imat_dataset
import argparse

# imports for segmentation
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

# imports external packages (from folder)
import bounding_box as bbx
import helpers
import utils
import transforms as T
from timm.models.layers import get_act_layer
from timm import create_model
import train
import yaml


parser = argparse.ArgumentParser(description='Training Config')

parser.add_argument('--chunk-size', type=int, default=20, metavar='CHUNK_SIZE',
                    help='H5PY chunk size (default: 20)')
parser.add_argument('--data-limit', type=int, default=12500, metavar='DATA_LIMIT',
                    help='Specify data limit, None to use all the data (default=12500)')
parser.add_argument('--target-dim', type=int, default=512, metavar='DIM',
                    help='Dimention of the images. It is vital that the image size will be devisiable by 2 at least 6 times (default=512)')
parser.add_argument('--delete-existing', type=bool, default=False, metavar='BOOL',
                    help='Delete existing H5PY files, if False will only add more data to the file (default=False)')


def parse_args():
    # parse the args that are passed to this script
    args = parser.parse_args()

    # save the args as a text string so we can log them later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


class DatasetH5Writer(torch.utils.data.Dataset):
    def __init__(self, dataset, target_dim, out_file, chunk_size, delete_existing):
        super(DatasetH5Writer, self).__init__()
        self.dataset = dataset
        self.dataset_len = self.dataset.__len__()
        self.chunk_size = chunk_size
        self.target_dim = target_dim
        self.file_name = out_file
        self.cpu_count = multiprocessing.cpu_count()
        self.delete_existing = delete_existing
        requires_init = False
        if os.path.exists(self.file_name):
            if self.delete_existing:
                os.remove(self.file_name)
                requires_init
        else:
            requires_init
        
        self.h5py_file = h5py.File(self.file_name, "a")
        if requires_init:
            self.image_ids_data_set = self.h5py_file.create_dataset("image_ids", shape=(0,), dtype=np.uint64, maxshape=(None,), chunks=(self.chunk_size,))
            self.images_data_set = self.h5py_file.create_dataset("images", shape=(0,3,self.target_dim,self.target_dim), dtype=np.float64, maxshape=(None,3,self.target_dim,self.target_dim), chunks=(self.chunk_size,3,self.target_dim,self.target_dim))
            dt = h5py.vlen_dtype(np.dtype('int64'))
            self.labels_data_set = self.h5py_file.create_dataset("labels", shape=(0,), maxshape=(None,), dtype=dt, chunks=(self.chunk_size,))
            self.masks_data_set = self.h5py_file.create_dataset("masks", shape=(0,75,self.target_dim,self.target_dim), maxshape=(None,75,512,512), dtype=np.uint8, chunks=(self.chunk_size,75,self.target_dim,self.target_dim))
            self.boxes_data_set = self.h5py_file.create_dataset("boxes", shape=(0,75,4), maxshape=(None,75,4), dtype=np.float64, chunks=(self.chunk_size,75,4))
        else:
            self.image_ids_data_set = self.h5py_file['image_ids']
            self.images_data_set = self.h5py_file['images']
            self.labels_data_set = self.h5py_file['labels']
            self.masks_data_set = self.h5py_file['masks']
            self.boxes_data_set = self.h5py_file['boxes']

        self.start_idx = self.images_data_set.shape[0]
        if self.start_idx != 0:
            assert self.target_dim == self.images_data_set.shape[-2]

        assert self.start_idx not in self.image_ids_data_set

    def append_to_h5py(self, result):
        chunk_size, images_np, image_ids_np, labels_numpy_list, masks_numpy_fixed_size, boxes_numpy_fixed_size = result
        assert images_np.shape[0] == chunk_size
        assert len(masks_numpy_fixed_size) == chunk_size
        assert len(image_ids_np) == chunk_size

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
    def process_chunk(dataset, start_idx, chunk_size, target_dim):
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

    def process(self, debug=False):
        print("CPU count is [{}]".format(self.cpu_count))
        print("Started writing [{}]...".format(self.file_name))
        pool = multiprocessing.Pool(self.cpu_count - 1)
        queue = multiprocessing.Queue()
        idx = self.start_idx
        results = []
        count_chunks = 0
        while idx < self.dataset_len:
            res = pool.apply_async(DatasetH5Writer.process_chunk, (self.dataset, idx, self.chunk_size, self.target_dim), callback=queue.put)
            idx += self.chunk_size
            count_chunks += 1
            if debug:
                results.append(res)
        if debug:
            for res in results:
                res.wait()
                print(res)
                print(res.get())
        pool.close()
    
        count_chunks_done = 0
        while count_chunks > 0 and True:
            if not queue.empty():
                print("Approx. queue size [{}]".format(queue.qsize()))
                count_chunks_done += 1
                self.append_to_h5py(queue.get())
                print("Processed chunks [{}/{}]".format(count_chunks_done, count_chunks))
                if count_chunks_done == count_chunks:
                    break
                    
        pool.join()
        assert len(self.image_ids_data_set) == len(np.unique(self.image_ids_data_set)), "Some index was written twice"
        print("File [{}] completed.".format(self.file_name))
            
    def close(self):
        self.h5py_file.close()


def main():
    args, args_text = parse_args()
    print("Args: {}".format(args_text))

    main_folder_path = '../'
    num_classes, train_df, test_df, categories_df = train.process_data(main_folder_path, args.data_limit)

    dataset_test = imat_dataset.IMATDataset(main_folder_path, test_df, num_classes, args.target_dim, "effdet", False, T.get_transform(train=False), gather_statistics=False)
    h5_test_writer = DatasetH5Writer(dataset_test, args.target_dim, "../imaterialist_test_" + str(args.target_dim) + ".hdf5", chunk_size=args.chunk_size, delete_existing=args.delete_existing)
    h5_test_writer.process()
    h5_test_writer.close()

    dataset = imat_dataset.IMATDataset(main_folder_path, train_df, num_classes, args.target_dim, False, "effdet", T.get_transform(train=False), gather_statistics=False)
    h5_writer = DatasetH5Writer(dataset, args.target_dim, "../imaterialist_" + str(args.target_dim) + ".hdf5", chunk_size=args.chunk_size, delete_existing=args.delete_existing)
    h5_writer.process()
    h5_writer.close()

    print("All done.")

if __name__ == '__main__':
    main()