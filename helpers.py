import torch
from PIL import Image
import torchvision.transforms as transforms


def rescale(matrix, target_dim, pad_color=0, interpolation=Image.NEAREST):
    mode = None
    if isinstance(matrix, Image.Image):
        mode = 'RGB'
        matrix_img = matrix.copy()
    else:
        mode = 'L'  # Luminance, single channel
        matrix_img = transforms.ToPILImage(mode=mode)(matrix.clone())

    if target_dim:
        orig_shape = matrix_img.size  # old_size[0] is in (width, height) format

        ratio = float(target_dim)/max(orig_shape)
        new_size = tuple([int(x*ratio) for x in orig_shape])
        # thumbnail is a in-place operation
        matrix_img.thumbnail(new_size, resample=interpolation)

        # create a new image and paste the resized image on it
        new_im = Image.new(mode, (target_dim, target_dim))
        new_im.paste(matrix_img, ((target_dim-new_size[0])//2, (target_dim-new_size[1])//2))
    else:
        new_im = matrix_img
    
    trans = transforms.ToTensor()
    if isinstance(matrix, Image.Image):
        return trans(new_im)
    else:
        return trans(new_im) * 255  # TODO(ofekp): note that masks will get this


def get_labels(image_df):
    return torch.as_tensor(list(image_df['ClassId']), dtype=torch.int64)


def get_masks(image_df, target_dim=None):
    '''
    given: the image df from the train_df or test_df
    return: binary masks as a list
    '''
    segments = list(image_df['EncodedPixels'])
    class_ids = list(image_df['ClassId'])
    
    height = image_df['Height'][0]
    width = image_df['Width'][0]
    masks = []

    for segment, class_id in zip(segments, class_ids):
        # initialize empty mask
        mask = torch.zeros((height, width), dtype=torch.uint8).reshape(-1)

        # iterate over encoded pixels to create the mask for this segment
        split_pixels = list(map(int, segment.split()))
        pixel_starts = split_pixels[::2]
        run_lengths = split_pixels[1::2]
        assert max(pixel_starts) < mask.shape[0]
        for pixel_start, run_length in zip(pixel_starts, run_lengths):
            pixel_start = int(pixel_start) - 1
            run_length = int(run_length)
            mask[pixel_start:pixel_start + run_length] = 1

        # TODO(ofekp): try to find how to do this without first converting to numpy
        mask = torch.tensor(mask.numpy().reshape(height, width, order='F'), dtype=torch.uint8)
        mask = rescale(mask, target_dim).type(torch.ByteTensor)  # masks should have dtype=torch.uint8
        mask = mask.squeeze()
        masks.append(mask)

    # there is a chance that the mask will be all zeros after the rescale if the object was too small
    # do not skip inserting bad masks, they will be filtered lated by remove_empty_masks
    count_bad = 0
    for mask in masks:
        assert torch.min(mask) >= 0
        assert torch.max(mask) <= 1
        if torch.max(mask) != 1.0:
            count_bad += 1
    if count_bad > 0:
        image_id = image_df['ImageId'].iloc[0]
        #print('ERROR: Image [{}] contains [{}] segments that were erased due to rescaling with target_dim [{}]'.format(image_id, count_bad, target_dim))
    return torch.stack(masks)


def get_bounding_boxes(image_df, masks):
    bounding_boxes = []
    for curr_mask in masks:  # [512, 512, 3]
        x_left = 0.0
        y_top = 0.0
        x_right = 0.0
        y_bottom = 0.0
        if torch.max(curr_mask) == 1.0:
            rows_sum = torch.sum(curr_mask, axis=1)
            rows_sum_non_zero = torch.where(rows_sum > 0)[0]
            y_top = torch.min(rows_sum_non_zero)
            y_bottom = torch.max(rows_sum_non_zero)
    
            cols_sum = torch.sum(curr_mask, axis=0)
            cols_sum_non_zero = torch.where(cols_sum > 0)[0]
            x_left = torch.min(cols_sum_non_zero)
            x_right = torch.max(cols_sum_non_zero)
        
            # order is important as it is the input order the bb package is expecting,
            # so do NOT skip inserting the image
            if not (x_left >= 0 and x_left < x_right and y_top >= 0 and y_bottom > y_top):
                image_id = image_df['ImageId'].iloc[0]
                #print("ERROR: Image [{}] x_left [{}] y_top [{}] x_right [{}] y_bottom [{}]".format(image_id, str(x_left), str(y_top), str(x_right), str(y_bottom)))
            bounding_boxes.append((x_left, y_top, x_right, y_bottom))
        
    return torch.as_tensor(bounding_boxes, dtype=torch.float32)


def remove_empty_masks(labels, masks, bounding_boxes):
    indices_to_keep_masks = []  # empty array with 1 dim
    idx = 0
    for mask in masks:
        if torch.max(mask).cpu().numpy() == 1:
            indices_to_keep_masks.append(idx)
        idx += 1

    indices_to_keep_bbx = []  # empty array with 1 dim
    idx = 0
    for box in bounding_boxes:
        if ((box[3] - box[1]) > 0) and ((box[2] - box[0]) > 0):
            indices_to_keep_bbx.append(idx)
        idx += 1

    # TODO(ofekp): remove segments that have an area less than some thresholds (e.g. image id '361cc7654672860b1b7c85fe8e92b38a')

    indices_to_keep = torch.tensor(list(set(indices_to_keep_masks) & set(indices_to_keep_bbx)), dtype=int)
    return labels[indices_to_keep], masks[indices_to_keep], bounding_boxes[indices_to_keep]


# def worker(q, lock, counter, x):
#     time.sleep(3.0 / x)
#     q.put(x*x)
#     lock.acquire()
#     try:
#         counter.value += 1
#     finally:
#         lock.release()

# def inc_counter(lock, counter):
#     lock.acquire()
#     try:
#         counter.value += 1
#     finally:
#         lock.release()
#     print(counter.value)


# def worker(processed_data_folder_path, worker_id, image_ids, train_df, lock, counter, skipped_images):
#     train_processed_df = pd.DataFrame()
#     for image_id in image_ids:
#         vis_df = train_df[train_df['ImageId'] == image_id]
#         vis_df = vis_df.reset_index(drop=True)
#         labels = get_labels(vis_df)
#         try:
#             masks = get_masks(vis_df)
#         except:
#             skipped_images.put(image_id)
#             inc_counter(lock, counter)
#             continue
#         boxes = get_bounding_boxes(masks)
#         image_df = pd.DataFrame({"image_id": image_id, "labels": [labels], "masks": [masks], "boxes": [boxes]})
#         train_processed_df = train_processed_df.append(image_df)
#         inc_counter(lock, counter)
#     train_processed_df.to_pickle(processed_data_folder_path + str(worker_id))
#     # q.put(train_processed_df)
#     # q.close()
#     # skipped_images.close()
#     return
