import bounding_box as bbx
import numpy as np
import helpers
from PIL import Image
from datetime import datetime
import os
import torch
import matplotlib.pyplot as plt
import imat_dataset
import common
import math

font = bbx.get_font_with_size(10)

class Visualize:
    def __init__(self, main_folder_path, categories_df, target_dim, dest_folder=None):
        self.main_folder_path = main_folder_path
        self.target_dim = target_dim
        self.categories_df = categories_df
        self.dest_folder = dest_folder


    # generate a map from the class id to the label
    def get_label(self, class_id, allowed_classes=None):
        if allowed_classes is not None:
            class_id = allowed_classes[class_id]
        class_name = self.categories_df.loc[class_id]['name']
        label = class_name
        if ',' in class_name:
            label = class_name.split(',')[0]
        label = "(" + str(class_id) + ") " + label
        return label


    # bounding_boxes - xyxy format
    def get_image_bounding_boxes(self, height, width, bounding_boxes, labels, decode_labels=True):
        if self.target_dim == None:
            image_with_bb = np.zeros((height, width, 3))
        else:
            image_with_bb = np.zeros((self.target_dim, self.target_dim, 3))
        
        for box, class_id in zip(bounding_boxes, labels):
            class_id = class_id.cpu().numpy()
            bbx.add(image_with_bb, *box, label=(self.get_label(class_id) if decode_labels else str(class_id)), font=font)

        # creating the alpha channel for the bounding box image (to avoid obscuring the original image with black background)
        bb_alpha = np.array(np.max(image_with_bb[:,:,:] > 0, axis=2) > 0, dtype=int)
        bb_alpha = bb_alpha * 255
        image_with_bb = np.concatenate((image_with_bb, bb_alpha.reshape(bb_alpha.shape[0], bb_alpha.shape[1], 1)), axis=2)
        image_with_bb = np.array(image_with_bb, dtype=int)
        return image_with_bb


    def show_image_data_ground_truth(self, data_df, image_id, is_colab, figsize=(40, 40)):
        # Get the an image id given in the training set for visualization
        vis_df = data_df[data_df['ImageId'] == image_id]
        vis_df = vis_df.reset_index(drop=True)
        class_ids = helpers.get_labels(vis_df)
        masks = helpers.get_masks(vis_df, target_dim=self.target_dim)
        bounding_boxes = helpers.get_bounding_boxes(vis_df, masks)
        class_ids, masks, bounding_boxes = helpers.remove_empty_masks(class_ids, masks, bounding_boxes)
        img = Image.open(common.get_image_path(self.main_folder_path, image_id, is_colab)).convert("RGB")
        img = helpers.rescale(img, target_dim=self.target_dim)
        self.show_image_data(img, class_ids, masks, bounding_boxes, figsize=figsize)


    def show_image_data(self, img, class_ids, masks, bounding_boxes, figsize=(40, 40), split_segments=False, grid_layout=False):
        height = img.shape[2]
        width = img.shape[1]
        image_with_bb = self.get_image_bounding_boxes(height, width, bounding_boxes, class_ids)
        
        if self.target_dim == None:
            mask = torch.zeros((height, width))
        else:
            mask = torch.zeros((self.target_dim, self.target_dim))
        
        # generate the segments mask with colors
        if masks is not None:
            for i, (curr_mask, class_id) in enumerate(zip(masks, class_ids)):
                curr_mask = curr_mask.cpu()
                class_id = class_id.cpu()
                assert torch.min(curr_mask) >= 0.0
                assert torch.max(curr_mask) <= 1.0
                curr_mask = curr_mask.type(torch.FloatTensor)
                mask = torch.where(curr_mask == 0, mask, curr_mask * (255 - 4 * class_id))

        if not split_segments:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
            ax[0].imshow(img.permute(1, 2, 0))
            ax[0].axis('off')
            ax[1].imshow(img.permute(1, 2, 0))
            ax[1].imshow(mask, alpha=0.7)
            ax[1].axis('off')
            ax[2].imshow(img.permute(1, 2, 0))
            ax[2].imshow(image_with_bb)
            ax[2].axis('off')
        else:
            num_segments = len(masks)
            if grid_layout:
                fig, ax = plt.subplots(nrows=int(math.ceil((num_segments+2) / 3)), ncols=3, figsize=figsize)
                i = 0
                ax[0, 0].imshow(img.permute(1, 2, 0))
                ax[0, 0].axis('off')
                i += 1
                r = int(i / 3)
                c = i % 3
                for curr_mask, class_id in zip(masks, class_ids):
                    curr_mask = curr_mask.cpu()
                    class_id = class_id.cpu()
                    assert torch.min(curr_mask) >= 0.0
                    assert torch.max(curr_mask) <= 1.0
                    curr_mask = curr_mask.type(torch.FloatTensor)
                    ax[r, c].imshow(img.permute(1, 2, 0))
                    ax[r, c].imshow(curr_mask, alpha=0.7)
                    ax[r, c].axis('off')
                    i += 1
                    r = int(i / 3)
                    c = i % 3
                ax[r, c].imshow(img.permute(1, 2, 0))
                ax[r, c].imshow(image_with_bb)
                ax[r, c].axis('off')
            else:
                fig, ax = plt.subplots(nrows=1, ncols=(num_segments+2), figsize=figsize)
                i = 0
                ax[i].imshow(img.permute(1, 2, 0))
                ax[i].axis('off')
                i += 1
                i = i
                for curr_mask, class_id in zip(masks, class_ids):
                    curr_mask = curr_mask.cpu()
                    class_id = class_id.cpu()
                    assert torch.min(curr_mask) >= 0.0
                    assert torch.max(curr_mask) <= 1.0
                    curr_mask = curr_mask.type(torch.FloatTensor)
                    ax[i].imshow(img.permute(1, 2, 0))
                    ax[i].imshow(curr_mask, alpha=0.7)
                    ax[i].axis('off')
                    i += 1
                ax[i].imshow(img.permute(1, 2, 0))
                ax[i].imshow(image_with_bb)
                ax[i].axis('off')

        if self.dest_folder is None:
            plt.show()
        else:
            if not os.path.exists(self.dest_folder):
                os.mkdir(self.dest_folder)
            plt.savefig(self.dest_folder + "/" + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + '.png')


    def show_prediction_on_img(self, model, dataset, dataset_df, img_idx, is_colab, show_groud_truth=True, box_threshold=0.001, split_segments=False, grid_layout=False):
        if isinstance(dataset, imat_dataset.IMATDatasetH5PY):
            img, _ = dataset.__getitem__(img_idx)
        else:
            img, _ = dataset[img_idx]
        # img_formatted = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()

        # put the model in evaluation mode
        with torch.no_grad():
            device = next(model.parameters()).device
            if box_threshold is not None:
                prediction = model([img.to(device)], box_threshold=box_threshold)
            else:
                # case of faster rcnn model
                prediction = model([img.to(device)])

        # class_ids, masks, boxes = helpers.remove_empty_masks(class_ids, masks, boxes)
        boxes = prediction[0]['boxes']
        class_ids = prediction[0]['labels']
        masks = prediction[0]['masks'][:, 0]
        if show_groud_truth:
            if isinstance(dataset, imat_dataset.IMATDatasetH5PY):
                image_ids = dataset_df['ImageId'].unique()
                image_id = dataset.dataset_h5py_reader.get_image_id(img_idx)
                print(image_id)
                self.show_image_data_ground_truth(dataset_df, image_ids[image_id], is_colab)
            else:
                self.show_image_data_ground_truth(dataset_df, dataset.image_ids[img_idx], is_colab)
        self.show_image_data(img, class_ids, masks, boxes, split_segments=split_segments, grid_layout=grid_layout)