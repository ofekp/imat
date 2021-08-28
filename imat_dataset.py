import time
import multiprocessing
from multiprocessing import Process, Pool, Queue, Lock, Value
import helpers
import torch
import h5py
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
import common


class IMATDataset(BaseDataset):
    def __init__(self, main_folder_path, data_df, num_classes, target_dim, model_name, is_colab, transforms=None, gather_statistics=True):
        self.main_folder_path = main_folder_path
        self.data_df = data_df
        self.num_classes = num_classes
        self.target_dim = target_dim
        self.is_colab = is_colab
        self.transforms = transforms
        self.model_name = model_name
        self.image_ids = data_df['ImageId'].unique()
        # TODO: indices = torch.randperm(len(dataset)).tolist()
        self.skipped_images = []
        self.gather_statistics = gather_statistics
        if self.gather_statistics:
            self.lock = Lock()
            self.images_processed = Value('i', 0)
            self.total_transform_time = Value('f', 0.0)
            self.total_mask_time = Value('f', 0.0)
            self.total_box_time = Value('f', 0.0)
            self.total_process_time = Value('f', 0.0)
            self.total_image_load_time = Value('f', 0.0)

    def inc_by(self, lock, counter, val):
        lock.acquire()
        try:
            counter.value += val
        finally:
            lock.release()
        
    def show_stats(self):
        images_processed = self.images_processed.value
        total_process_time = self.total_process_time.value
        avg_time_per_image = 0 if images_processed == 0 else total_process_time / images_processed
        avg_image_load_time = 0 if images_processed == 0 else self.total_image_load_time.value / images_processed
        avg_transform_time = 0 if images_processed == 0 else self.total_transform_time.value / images_processed
        avg_mask_time = 0 if images_processed == 0 else self.total_mask_time.value / images_processed
        avg_box_time = 0 if images_processed == 0 else self.total_box_time.value / images_processed
        print("Processed [{}] images in [{}] seconds"
              " Avg per image [{}]"
              " Avg image load time [{}]"
              " Avg transform time [{}]"
              " Avg mask time [{}]"
              " Avg box time [{}]"
              .format(
            images_processed,
            total_process_time,
            avg_time_per_image,
            avg_image_load_time,
            avg_transform_time,
            avg_mask_time,
            avg_box_time))
    
    def __getitem__(self, idx):
        if self.gather_statistics:
            start = time.time()
        image_id = self.image_ids[idx]
        vis_df = self.data_df[self.data_df['ImageId'] == image_id]
        vis_df = vis_df.reset_index(drop=True)
        labels = helpers.get_labels(vis_df)
        mask_start_ts = time.time()
        try:
            masks = helpers.get_masks(vis_df, target_dim=self.target_dim)
            for mask in masks:
                assert not torch.any(torch.isnan(mask))
                assert torch.where(mask > 0)[0].shape[0] == torch.sum(mask)  # check only ones and zeros
        except Exception as e:
            self.skipped_images.append(image_id)
            print("ERROR: Skipped image with id [{}] due to a mask exception [{}]".format(image_id, e))
            return
        if self.gather_statistics:
            self.inc_by(self.lock, self.total_mask_time, time.time() - mask_start_ts)
        
        num_objs = len(labels)

        image_id_idx = idx
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        labels, masks = helpers.remove_empty_masks(labels, masks)

        box_start_ts = time.time()
        boxes = helpers.get_bounding_boxes(masks)
        labels, boxes, masks = helpers.remove_empty_boxes(labels, boxes, masks)
        try:
            for box in boxes:
                assert not torch.any(torch.isnan(box))
        except Exception as e:
            self.skipped_images.append(image_id)
            print("ERROR: Skipped image with id [{}] due to a BB exception [{}]".format(image_id, e))
            return
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        if self.gather_statistics:
            self.inc_by(self.lock, self.total_box_time, time.time() - box_start_ts)

        target = {}
        if "faster" in self.model_name:
            target["labels"] = labels
            assert torch.min(target["labels"]) >= 0
            assert torch.max(target["labels"]) <= self.num_classes - 1
        else:
            # we only need the correction for the modified model
            target["labels"] = torch.add(labels, 1)  # refer to fast_collate, this is needed for efficient det
            assert torch.min(target["labels"]) >= 1
            assert torch.max(target["labels"]) <= self.num_classes
        target["masks"] = masks
        target["boxes"] = boxes
        target["image_id"] = image_id_idx
        target["area"] = area
        target["iscrowd"] = iscrowd
#         target["image_id"] = torch.tensor(image_id_idx)
#         target["area"] = torch.tensor(area)
#         target["iscrowd"] = torch.tensor(iscrowd)

        image_load_start_ts = time.time()
        image_orig = Image.open(common.get_image_path(self.main_folder_path, image_id, self.is_colab)).convert("RGB")
        image = helpers.rescale(image_orig, target_dim=self.target_dim)
        if self.gather_statistics:
            self.inc_by(self.lock, self.total_image_load_time, time.time() - image_load_start_ts)
        
        # TODO(ofekp): check what happens here when the image is < self.target_dim. What will helpers.py scale method do to the image in this case?
        target["img_size"] = image_orig.size[-2:] if self.target_dim is None else (self.target_dim, self.target_dim)
        image_orig_max_dim = max(target["img_size"])
        img_scale = self.target_dim / image_orig_max_dim
        target["img_scale"] = 1. / img_scale  # back to original size
        
        if self.gather_statistics:
            transform_start_ts = time.time()
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        if self.gather_statistics:
            self.inc_by(self.lock, self.total_transform_time, time.time() - transform_start_ts)
            self.inc_by(self.lock, self.images_processed, 1)
            self.inc_by(self.lock, self.total_process_time, time.time() - start)
        
        assert image.shape[0] <= self.target_dim and image.shape[1] <= self.target_dim and image.shape[2] <= self.target_dim
        return image, target

    def __len__(self):
        return len(self.image_ids)


class DatasetH5Reader(torch.utils.data.Dataset):
    def __init__(self, in_file):
        super(DatasetH5Reader, self).__init__()
        self.in_file = in_file

    def __getitem__(self, index):
        h5py_file = h5py.File(self.in_file, "r", swmr=True)  # swmr=True allows concurrent reads
        image = h5py_file['images'][index]
        labels = h5py_file['labels'][index]
        masks_fixed_size = h5py_file['masks'][index]
        boxes_fixed_size = h5py_file['boxes'][index]
        return image, labels, masks_fixed_size, boxes_fixed_size
    
    def get_image_id(self, idx):
        '''
        Images in the h5py dataset are not inserted in their order according to how they appear in dataframe
        this is why a translation has to be made to grab the correct image
        '''
        h5py_file = h5py.File(self.in_file, "r", swmr=True)  # swmr=True allows concurrent reads
        image_idxes = h5py_file['image_ids'][:]
        h5py_file.close()
        print(image_idxes[0:50])
        return image_idxes[idx]

    def __len__(self):
        h5py_file = h5py.File(self.in_file, "r", swmr=True)  # swmr=True allows concurrent reads
        return h5py_file['images'].shape[0]


class IMATDatasetH5PY(BaseDataset):
    def __init__(self, dataset_h5py_reader, num_classes, target_dim, model_name, transforms=None):
        self.transforms = transforms
        self.num_classes = num_classes
        self.target_dim = target_dim
        self.model_name = model_name
        self.lock = Lock()
        self.dataset_h5py_reader = dataset_h5py_reader
        self.images_processed = Value('i', 0)
        self.total_process_time = Value('f', 0.0)
        # TODO: indices = torch.randperm(len(dataset)).tolist()

    def inc_by(self, lock, counter, val):
        lock.acquire()
        try:
            counter.value += val
        finally:
            lock.release()
        
    def show_stats(self):
        images_processed = self.images_processed.value
        total_process_time = self.total_process_time.value
        avg_time_per_image = 0 if images_processed == 0 else total_process_time / images_processed
        print("Processed [{}] images in [{}] seconds"
              " Avg per image [{}]"
              .format(
            images_processed,
            total_process_time,
            avg_time_per_image))
    
    def __getitem__(self, idx):
        start = time.time()
        
        # it is critical to open the file here and not in the CTOR, to avoid errors on multiple access of threads
        # the error I got was: "OSError: Can't read some data (inflate() failed) & (wrong B-tree signature)"
        image, labels, masks, boxes = self.dataset_h5py_reader.__getitem__(idx)
        image = torch.from_numpy(image).float()
        target = {}
        if len(labels) == 0:
            print("idx [{}] had an image with 0 labels".format(idx))
        assert len(labels) > 0
        target["labels"] = torch.from_numpy(labels)
        assert torch.min(target["labels"]) >= 1
        assert torch.max(target["labels"]) <= self.num_classes
        if "faster" in self.model_name:
            # in case of the conventional model, we need to have the classes start from 0
            target["labels"] = torch.sub(target["labels"], 1)
        
        num_objs = target["labels"].shape[0]
        target["masks"] = torch.from_numpy(masks[0:num_objs]).type(torch.uint8)
        target["boxes"] = torch.from_numpy(boxes[0:num_objs]).float()
        target["image_id"] = torch.tensor([idx])
        area = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        target["area"] = area  # TODO(ofekp): where is this being used and should it be a tensor?
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target["iscrowd"] = iscrowd
        
        # TODO(ofekp): check what happens here when the image is < self.target_dim. What will helpers.py scale method do to the image in this case?
        target["img_size"] = (self.target_dim, self.target_dim)
        image_orig_max_dim = max(target["img_size"])
        img_scale = self.target_dim / image_orig_max_dim
        target["img_scale"] = 1. / img_scale  # back to original size
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        self.inc_by(self.lock, self.images_processed, 1)
        self.inc_by(self.lock, self.total_process_time, time.time() - start)
        return image, target
        

    def __len__(self):
        return self.dataset_h5py_reader.__len__()