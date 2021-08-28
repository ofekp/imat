import math
import sys
import time
import torch
import psutil
import torchvision.models.detection.mask_rcnn
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import gc
from memory_profiler import profile


def train_one_epoch(model, optimizer, data_loader, device, epoch, gradient_accumulation_steps, print_freq, box_threshold):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    optimizer.zero_grad()  # gradient_accumulation
    steps = 0  # gradient_accumulation
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # print("target: {}".format(targets))

        steps += 1  # gradient_accumulation
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        if box_threshold is None:
            loss_dict = model(images, targets)
        else:
            loss_dict = model(images, box_threshold, targets)

        # print(loss_dict, flush=True)
        
        losses = sum(loss / gradient_accumulation_steps for loss in loss_dict.values())  # gradient_accumulation

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), flush=True)
            print(loss_dict_reduced)
            sys.exit(1)

        #optimizer.zero_grad()
        losses.backward()

        # ofekp: we add grad clipping here to avoid instabilities in training
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        # gradient_accumulation
        if steps % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def evaluate(model, data_loader, device, box_threshold=0.1):
    with torch.no_grad():
        # mu = psutil.virtual_memory().percent
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        # print("Memory usage 1 [{}]".format(psutil.virtual_memory().percent), flush=True)
        coco = get_coco_api_from_dataset(data_loader.dataset, box_threshold)
        # print("Memory usage 2 [{}]".format(psutil.virtual_memory().percent), flush=True)
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        # diff_setup = psutil.virtual_memory().percent - mu

        # print("Memory usage 3 [{}]".format(psutil.virtual_memory().percent), flush=True)
        # accu_diff_iter = 0
        count = 0
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            one_iteration(device, images, box_threshold, model, cpu_device, targets, coco_evaluator, metric_logger)
            count += 1
            if count % 20 == 0:
                print("Memory usage [{}]".format(psutil.virtual_memory().percent), flush=True)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        del coco
        del coco_evaluator
        gc.collect()
        print("Memory usage after training one epoch [{}]".format(psutil.virtual_memory().percent), flush=True)
    # print("Diff in memory for setup is [{}] Diff in memory for eval is [{}] accu_diff_iter [{}]".format(diff_setup, psutil.virtual_memory().percent - mu, accu_diff_iter), flush=True)
    return None


def one_iteration(device, images, box_threshold, model, cpu_device, targets, coco_evaluator, metric_logger):
    images_gpu = list(img.to(device) for img in images)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_time = time.time()
    if box_threshold is None:
        outputs = model(images_gpu)
    else:
        outputs = model(images_gpu, box_threshold)


    # targets_cpu = [{k: v.to(cpu_device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    model_time = time.time() - model_time

    res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}  # ofekp: this used to be target["image_id"].item()
    evaluator_time = time.time()
    coco_evaluator.update(res)
    evaluator_time = time.time() - evaluator_time
    metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    if psutil.virtual_memory().percent > 50.0:
        print("Memory usage too high! Exiting!")
        return None
    
    # print("---");

    # gc.collect()
    # for image_cpu in images_gpu:
    #     del image_cpu
    # for output in outputs:
    #     for k in list(output.keys()):
    #         del output[k]
    # for target_cpu in targets_cpu:
    #     for k in list(target_cpu.keys()):
    #         del target_cpu[k]
