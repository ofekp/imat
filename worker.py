import time 
import pandas as pd
import numpy as np
import cv2
import torch

# def worker(q, lock, counter, x):
#     time.sleep(3.0 / x)
#     q.put(x*x)
#     lock.acquire()
#     try:
#         counter.value += 1
#     finally:
#         lock.release()

# INTER_CUBIC
def rescale(matrix, target_dim, pad_color=0, interpolation=cv2.INTER_NEAREST):
    width = matrix.shape[1]
    height = matrix.shape[0]
    aspect_ratio = width / height
    if target_dim == None or (width == target_dim and height == target_dim):
        return matrix

    if width >= height:  # aspect_ratio >= 1
        new_width = target_dim
        new_height = int(new_width // aspect_ratio)
        pad_left = pad_right = 0
        pad_top = pad_bottom = (target_dim - new_height) // 2
        if (pad_top * 2 + new_height) < target_dim:
            pad_top += 1
        assert (pad_top + pad_bottom + new_height) == target_dim
    else:  # aspect_ratio < 1
        new_height = target_dim
        new_width = int(new_height * aspect_ratio)
        pad_left = pad_right = (target_dim - new_width) // 2
        pad_top = pad_bottom = 0
        if (pad_left * 2 + new_width) < target_dim:
            pad_left += 1
        assert (pad_left + pad_right + new_width) == target_dim
    
    matrix = cv2.resize(matrix, dsize=(new_width, new_height), interpolation=interpolation)
    matrix = cv2.copyMakeBorder(matrix, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return matrix


def get_labels(image_df):
    return torch.as_tensor(list(image_df['ClassId']), dtype=torch.int64)


def get_masks(image_df, target_dim=None, dtype=int):
    '''
    given: the image df from the train_df or test_df
    return: binary masks as a list (user can choose bool or int as dtype for the output)
    '''
    segments = list(image_df['EncodedPixels'])
    class_ids = list(image_df['ClassId'])
    
    height = image_df['Height'][0]
    width = image_df['Width'][0]
    masks = []

    for segment, class_id in zip(segments, class_ids):
        # initialize empty mask
        mask = np.zeros((height, width)).reshape(-1)

        # iterate over encoded pixels to create the mask for this segment
        splitted_pixels = list(map(int, segment.split()))
        pixel_starts = splitted_pixels[::2]
        run_lengths = splitted_pixels[1::2]
        assert max(pixel_starts) < mask.shape[0]
        for pixel_start, run_length in zip(pixel_starts, run_lengths):
            pixel_start = int(pixel_start) - 1
            run_length = int(run_length)
            mask[pixel_start:pixel_start + run_length] = 1

        mask = mask.reshape((height, width), order='F')
        mask = rescale(mask, target_dim)
        masks.append(np.array(mask, dtype=dtype))

    # there is a chance that the mask will be all zeros after the rescale
    count_bad = 0
    for mask in masks:
        if np.max(mask) != 1.0:
            count_bad += 1
    if count_bad > 0:
        image_id = image_df['ImageId'].iloc[0]
        print('ERROR: Image [{}] contains [{}] segments that were erased due to rescaling with target_dim [{}]'.format(image_id, count_bad, target_dim))
    return torch.as_tensor(masks, dtype=torch.uint8)


def get_bounding_boxes(image_df, masks):
    bounding_boxes = []
    for curr_mask in masks:
        x_left = 0.0
        y_top = 0.0
        x_right = 0.0
        y_bottom = 0.0
        if torch.max(curr_mask) == 1.0:
            rows_sum = torch.sum(curr_mask, axis=1)
            rows_sum_non_zero = torch.nonzero(rows_sum)
            y_top = torch.min(rows_sum_non_zero)
            y_bottom = torch.max(rows_sum_non_zero)
    
            cols_sum = torch.sum(curr_mask, axis=0)
            cols_sum_non_zero = torch.nonzero(cols_sum)
            x_left = torch.min(cols_sum_non_zero)
            x_right = torch.max(cols_sum_non_zero)
        
            # order is important as it is the input order the bb package is expecting
            if not (x_left >= 0 and x_left < x_right and y_top >= 0 and y_bottom > y_top):
                image_id = image_df['ImageId'].iloc[0]
                print("ERROR: Image [{}] x_left [{}] y_top [{}] x_right [{}] y_bottom [{}]".format(image_id, str(x_left), str(y_top), str(x_right), str(y_bottom)))
        bounding_boxes.append((x_left, y_top, x_right, y_bottom))
        
    return torch.as_tensor(bounding_boxes, dtype=torch.float32)


def inc_counter(lock, counter):
    lock.acquire()
    try:
        counter.value += 1
    finally:
        lock.release()
    print(counter.value)


def worker(processed_data_folder_path, worker_id, image_ids, train_df, lock, counter, skipped_images):
    train_processed_df = pd.DataFrame()
    for image_id in image_ids:
        vis_df = train_df[train_df['ImageId'] == image_id]
        vis_df = vis_df.reset_index(drop=True)
        labels = get_labels(vis_df)
        try:
            masks = get_masks(vis_df)
        except:
            skipped_images.put(image_id)
            inc_counter(lock, counter)
            continue
        boxes = get_bounding_boxes(masks)
        image_df = pd.DataFrame({"image_id": image_id, "labels": [labels], "masks": [masks], "boxes": [boxes]})
        train_processed_df = train_processed_df.append(image_df)
        inc_counter(lock, counter)
    train_processed_df.to_pickle(processed_data_folder_path + str(worker_id))
    # q.put(train_processed_df)
    # q.close()
    # skipped_images.close()
    return