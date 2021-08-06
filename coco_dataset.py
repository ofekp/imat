from torch.utils.data import Dataset as BaseDataset
from multiprocessing import Lock, Value
import time
import torch
from PIL import Image
import numpy as np
import helpers
import common


class COCODataset(BaseDataset):
    def __init__(self, coco, is_train, model_name, main_folder_path, target_dim, num_classes, transforms, is_colab, gather_statistics=True):
        self.main_folder_path = main_folder_path
        self.is_train = is_train
        self.coco = coco
        self.target_dim = target_dim
        self.gather_statistics = gather_statistics
        num_of_images_with_no_labels = len([id for id in self.coco.getImgIds() if len(self.coco.getAnnIds(imgIds=id, iscrowd=None)) == 0])
        print("Getting rid of [{}] images with no labels".format(num_of_images_with_no_labels), flush=True)
        self.image_ids = [id for id in self.coco.getImgIds() if len(self.coco.getAnnIds(imgIds=id, iscrowd=None)) > 0][:1000]
        print("Dataset [{}] contains [{}] images that have at least one instance".format("train" if self.is_train else "test", len(self.image_ids)), flush=True)
        self.model_name = model_name
        self.transforms = transforms
        self.num_classes = num_classes
        self.is_colab = is_colab
        if self.gather_statistics:
            self.lock = Lock()
            self.images_processed = Value('i', 0)
            self.total_process_time = Value('f', 0.0)

    def inc_by(self, lock, counter, val):
        lock.acquire()
        try:
            counter.value += val
        finally:
            lock.release()

    def show_stats(self):
        if not self.gather_statistics:
            return
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
        # image_orig = Image.open(
        #     common.get_image_path_coco(self.main_folder_path, self.is_train, img['file_name'], False)).convert("RGB")
        # return image_orig, {}
        start = time.time()
        image_id = self.image_ids[idx]
        img = self.coco.loadImgs(image_id)[0]
        annIds = self.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        labels = torch.as_tensor([ann['category_id'] for ann in anns], dtype=torch.int64)  # first label is 1
        if len(labels) == 0:
            raise Exception("Image with index [{}] has 0 labels as ground truth".format(idx))
        # bounding_boxes = []
        masks = []
        for ann in anns:
            # bb = ann['bbox']
            # x_left = bb[0]
            # y_top = bb[1]
            # x_right = x_left + bb[2]
            # y_bottom = y_top + bb[3]
            # bounding_boxes.append((x_left, y_top, x_right, y_bottom))
            orig_img_height = img['height']
            orig_img_width = img['width']
            mask = np.zeros((orig_img_height, orig_img_width))
            mask = np.maximum(self.coco.annToMask(ann), mask)
            assert mask.shape[0] == orig_img_height and mask.shape[1] == orig_img_width
            mask = torch.tensor(mask.reshape(orig_img_height, orig_img_width, order='F'), dtype=torch.uint8)
            assert mask.shape[0] == orig_img_height and mask.shape[1] == orig_img_width
            mask = helpers.rescale(mask, self.target_dim).reshape(self.target_dim, self.target_dim).type(torch.ByteTensor)  # masks should have dtype=torch.uint8
            assert mask.shape[0] == 512 and mask.shape[1] == 512
            mask = mask.squeeze()
            masks.append(mask)

        masks = torch.stack(masks)
        if len(labels) == 0:
            raise Exception("ERROR: Image with id [{}] has 0 labels as ground truth".format(image_id))

        labels, masks = helpers.remove_empty_masks(labels, masks)
        boxes = helpers.get_bounding_boxes(masks)
        labels, boxes, masks = helpers.remove_empty_boxes(labels, boxes, masks)
        if len(labels) == 0:
            raise Exception("ERROR: Image with id [{}] has 0 labels after removal of small objects".format(image_id))
        if not len(masks) == len(boxes) or not len(masks) == len(labels):
            raise Exception("ERROR: Mismatch for image with id [{}] after removing small objects, # masks [{}] # boxes [{}] # labels [{}]".format(image_id, len(masks), len(boxes), len(labels)))

        try:
            for box in boxes:
                assert not torch.any(torch.isnan(box))
        except Exception as e:
            raise Exception("ERROR: Skipped image with id [{}] due to a BB exception [{}]".format(image_id, e))

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        count_bad = 0
        for mask in masks:
            assert torch.min(mask) >= 0
            assert torch.max(mask) <= 1
            if torch.max(mask) != 1.0:
                count_bad += 1
        if count_bad > 0:
            raise Exception("ERROR: Found [{}/{}] bad masks".format(count_bad, len(masks)))

        target = {}
        if "faster" in self.model_name:
            target["labels"] = torch.sub(labels, 1)  # original model expects first label to be 0
            assert torch.min(target["labels"]) >= 0
            assert torch.max(target["labels"]) <= self.num_classes - 1
        else:
            target["labels"] = labels
            assert torch.min(target["labels"]) >= 1
            assert torch.max(target["labels"]) <= self.num_classes
        target["masks"] = masks
        target["boxes"] = boxes
        target["image_id"] = idx
        target["area"] = area
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        # image_orig = mpimg.imread(common.get_image_path_coco(self.images_folder, self.is_train, img['file_name'], False))
        image_orig = Image.open(common.get_image_path_coco(self.main_folder_path, self.is_train, img['file_name'], False)).convert("RGB")
        # image_orig = Image.Image()
        image = helpers.rescale(image_orig, target_dim=self.target_dim)
        assert image.shape[0] == 3 and image.shape[1] == 512 and image.shape[2] == 512

        # TODO(ofekp): check what happens here when the image is < self.target_dim. What will helpers.py scale method do to the image in this case?
        target["img_size"] = image_orig.size[-2:] if self.target_dim is None else (self.target_dim, self.target_dim)
        image_orig.close()
        image_orig_max_dim = max(target["img_size"])
        img_scale = self.target_dim / image_orig_max_dim
        target["img_scale"] = 1. / img_scale  # back to original size

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.gather_statistics:
            self.inc_by(self.lock, self.images_processed, 1)
            self.inc_by(self.lock, self.total_process_time, time.time() - start)

        # assert image.shape[0] <= self.target_dim and image.shape[1] <= self.target_dim and image.shape[2] <= self.target_dim
        assert image.shape[0] == 3 and image.shape[1] == 512 and image.shape[2] == 512
        assert len(masks) == len(boxes) and len(masks) == len(labels)
        return image, target

    def __len__(self):
        return len(self.image_ids)

    # # taken from https://towardsdatascience.com/how-to-analyze-the-coco-dataset-for-pose-estimation-7296e2ffb12e
    # def convert_to_df(self, coco):
    #     images_data = []
    #     persons_data = []
    #     # iterate over all images
    #     for img_id, img_fname, w, h, meta in get_meta(coco):
    #         images_data.append({
    #             'image_id': int(img_id),
    #             'path': img_fname,
    #             'width': int(w),
    #             'height': int(h)
    #         })
    #         # iterate over all metadata
    #         for m in meta:
    #             persons_data.append({
    #                 'image_id': m['image_id'],
    #                 'is_crowd': m['iscrowd'],
    #                 'bbox': m['bbox'],
    #                 'area': m['area'],
    #                 'num_keypoints': m['num_keypoints'],
    #                 'keypoints': m['keypoints'],
    #             })
    #     # create dataframe with image paths
    #     images_df = pd.DataFrame(images_data)
    #     images_df.set_index('image_id', inplace=True)
    #     # create dataframe with persons
    #     persons_df = pd.DataFrame(persons_data)
    #     persons_df.set_index('image_id', inplace=True)
    #     return images_df, persons_df

    # images_df, persons_df = convert_to_df(train_coco)
    # train_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
    # train_coco_df['source'] = 0
    #
    # images_df, persons_df = convert_to_df(val_coco)
    # val_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
    # val_coco_df['source'] = 1
    #
    # coco_df = pd.concat([train_coco_df, val_coco_df], ignore_index=True)

