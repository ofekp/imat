
def get_image_path(main_folder_path, image_id, is_colab):
    if is_colab:
        # in colab images are split into shards (folders) to avoid too many files in one folder
        shard_folder = image_id[0:2]
        return main_folder_path + f'/Data/shard/train/{shard_folder}/{image_id}.jpg'
    else:
        # non-sharded images
        return main_folder_path + f'/Data/train/{image_id}.jpg'


def get_image_path_coco(main_folder_path, is_train, image_file_name, is_colab):
    images_folder = 'train2017' if is_train else 'val2017'
    return main_folder_path + '/Data/{}/{}'.format(images_folder, image_file_name)
