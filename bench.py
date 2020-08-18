""" PyTorch EfficientDet support benches

Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
from timm.utils import ModelEma
from anchors import Anchors, AnchorLabeler, generate_detections, MAX_DETECTION_POINTS
from loss import DetectionLoss


def _post_process(config, cls_outputs, box_outputs):
    """Selects top-k predictions.

    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.

    Args:
        config: a parameter dictionary that includes `min_level`, `max_level`,  `batch_size`, and `num_classes`.

        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].

        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].
    """
    batch_size = cls_outputs[0].shape[0]
    cls_outputs_all = torch.cat([
        cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, config.num_classes])
        for level in range(config.num_levels)], 1)

    box_outputs_all = torch.cat([
        box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4])
        for level in range(config.num_levels)], 1)

    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=MAX_DETECTION_POINTS)
    indices_all = cls_topk_indices_all // config.num_classes
    classes_all = cls_topk_indices_all % config.num_classes

    box_outputs_all_after_topk = torch.gather(
        box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))

    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, config.num_classes))
    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))

    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all


@torch.jit.script
def _batch_detection(batch_size: int, class_out, box_out, anchor_boxes, indices, classes, img_scale, img_size):
    batch_detections = []
    # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
    for i in range(batch_size):
        detections = generate_detections(
            class_out[i], box_out[i], anchor_boxes, indices[i], classes[i], img_scale[i], img_size[i])
        batch_detections.append(detections)
    return torch.stack(batch_detections, dim=0)


class DetBenchPredict(nn.Module):
    def __init__(self, model, config):
        super(DetBenchPredict, self).__init__()
        self.config = config
        self.model = model
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)

    def forward(self, x, img_scales, img_size):
        class_out, box_out = self.model(x)
        class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)
        return _batch_detection(
            x.shape[0], class_out, box_out, self.anchors.boxes, indices, classes, img_scales, img_size)


# ofekp - adding this method from https://github.com/rwightman/efficientdet-pytorch/blob/0b36cc1cccfe92febc64f6eb569ca4393bd73964/data/loader.py#L86
import numpy as np 
def my_fast_collate(targets):
    MAX_NUM_INSTANCES = 150
    batch_size = len(targets)

    # FIXME this needs to be more robust
    target = dict()
    for k, v in targets[0].items():
        if torch.is_tensor(v):
            target_shape = (batch_size, MAX_NUM_INSTANCES)
            if len(v.shape) > 1:
                target_shape = (batch_size, MAX_NUM_INSTANCES) + v.shape[1:]
            target_dtype = v.dtype
        elif isinstance(v, np.ndarray):
            # if a numpy array, assume it relates to object instances, pad to MAX_NUM_INSTANCES
            target_shape = (batch_size, MAX_NUM_INSTANCES)
            if len(v.shape) > 1:
                target_shape = target_shape + v.shape[1:]
            target_dtype = torch.float32
        elif isinstance(v, (tuple, list)):
            # if tuple or list, assume per batch
            target_shape = (batch_size, len(v))
            target_dtype = torch.float32 if isinstance(v[0], float) else torch.int32
        else:
            # scalar, assume per batch
            target_shape = batch_size
            target_dtype = torch.float32 if isinstance(v, float) else torch.int64
        target[k] = torch.zeros(target_shape, dtype=target_dtype)

    for i in range(batch_size):
        for tk, tv in targets[i].items():
            if torch.is_tensor(tv):
                target[tk][i, 0:tv.shape[0]] = tv
            elif isinstance(tv, np.ndarray) and len(tv.shape):
                target[tk][i, 0:tv.shape[0]] = torch.from_numpy(tv)
            else:
                target[tk][i] = torch.tensor(tv, dtype=target[tk].dtype)

    return target


def my_fast_collate_images(images_list):
    batch_size = len(images_list)

    images_tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
    for i in range(batch_size):
        tensor[i] += torch.tensor(images_list[i] * 255, dtype=torch.uint8)
    return images_tensor


class DetBenchTrain(nn.Module):
    def __init__(self, model, config):
        super(DetBenchTrain, self).__init__()
        self.config = config
        self.model = model
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)
        self.anchor_labeler = AnchorLabeler(self.anchors, config.num_classes, match_threshold=0.5)
        self.loss_fn = DetectionLoss(self.config)

    def forward(self, images, features, targets):
        # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = images.tensors.device
        if targets is not None:
            target = my_fast_collate(targets)
            class_out, box_out = self.model(features)  # EfficientDetBB (without the FPN), expects to get the features (output of FPN)
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
                images.tensors.shape[0], target['boxes'].to(device), target['labels'].to(device))
            loss, class_loss, box_loss = self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)
            output = dict(loss=loss, class_loss=class_loss, box_loss=box_loss)
            if not self.training:
                # if eval mode, output detections for evaluation
                class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)
                output['detections'] = _batch_detection(
                    images.tensors.shape[0], class_out, box_out, self.anchors.boxes, indices, classes,
                    target['img_scale'].to(device), target['img_size'].to(device))
            return output
        else:
            # def forward(self, x, img_scales, img_size):
            class_out, box_out = self.model(features)
            class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)
            output = dict()
            # TODO(ofekp): 512 should use target_dim istead, can get this from images.image_sizes which is List[Tuple[int, int]]
            image_sizes = torch.tensor((512, 512)).repeat(images.tensors.shape[0], 1)
            output['detections'] = _batch_detection(
                images.tensors.shape[0], class_out, box_out, self.anchors.boxes, indices, classes, images.get_images_scales().to(device), image_sizes.to(device))
            return output

def unwrap_bench(model):
    # Unwrap a model in support bench so that various other fns can access the weights and attribs of the
    # underlying model directly
    if isinstance(model, ModelEma):  # unwrap ModelEma
        return unwrap_bench(model.ema)
    elif hasattr(model, 'module'):  # unwrap DDP
        return unwrap_bench(model.module)
    elif hasattr(model, 'model'):  # unwrap Bench -> model
        return unwrap_bench(model.model)
    else:
        return model
