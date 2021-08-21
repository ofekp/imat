# Code Modifications
For our work, we also modified the following repositories:
1. Main repository - https://github.com/ofekp/imat
1. EfficientDet modifications - https://github.com/ofekp/efficientdet-pytorch/ [[Code Comparison](https://github.com/rwightman/efficientdet-pytorch/compare/master...ofekp:master)]
1. Torchvision modifications - https://github.com/ofekp/vision [[Code Comparison](https://github.com/pytorch/vision/compare/master...ofekp:master)]

# Introduction
Recent advancements in the field of machine learning, specifically with problems such as instance segmentation and detection allow us to explore new combinations between the components that assemble those networks in order to create hybrid networks that improve on performance or results.
In this project we solve the competition of [iMaterialist](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/overview/description) using a modified Mask R-CNN making use of such recent advancements, EfficientNet as backbone, BiFPN and EfficientDet and achieve comparable or even improved results with smaller and faster models.

# Paper

Our project paper is available here:

[Fine_Grained_Segmentation_Task_for_Fashionpedia_Using_FusedEfficientDet_and_MaskRCNN](https://github.com/ofekp/imat/blob/master/Paper/Fine_Grained_Segmentation_Task_for_Fashionpedia_Using_FusedEfficientDet_and_MaskRCNN.pdf)

Our network architecture, refer to the project paper for detailed explanation:

![Project architecture](https://i.imgur.com/39SkxVL.png)

# Installation
pip install -r requirements.txt

# Creating a new H5PY dataset

While not a requirement, performing this step will greately improve training times.
If you wish to skip this step, remember to use `--h5py-dataset false` when training.

# Default setting

Please make note of the default settings, critically:

```
--data-limit 12500 - this is only a subset of the data on which we trained
--h5py-dataset true - means we use an H5PY dataset which requires pre-setup, see H5PY step
```

# Train a new model

```
nohup python train.py --load-model false --model-name tf_efficientdet_d0 --model-file-suffix effdet_d0 &
```

# Continue training saved model

```
nohup python train.py --load-model true --model-name tf_efficientdet_d0 --model-file-suffix effdet_d0 &
```

# Pre-trained Models

Can be found in [Releases](https://github.com/ofekp/imat/releases/)

# Visualization

Start jupyter note book
`nohup jupyter notebook --allow-root > jupyter_notebook.log`

Start `imat_visualization.ipynb` notebook

# Detailed Installation for GCP Instance

```
Starting up GCP env with TPU:
deploy deep learning host using this link
OS: Deep Learning on Linux, Debian GNU/Linux 9 Stretch PyTorhch/XLA (refer to https://github.com/pytorch/xla/issues/1666#issuecomment-589219327)
SSD 500
Allow full access to all Cloud APIs
Allow http, https

# Common to GPU and TPU
sudo apt-get update
TPU:
	conda activate /anaconda3/envs/torch-xla-1.6
GPU: 
	conda create --name imat; conda activate imat
conda install nb_conda  # this will add conda capabilities to jupyter notebooks
mkdir Project
cd Project
mkdir Data
git clone https://github.com/ofekp/imat.git Code
cd ..
ln -s Project/Code code
cd Project
export KAGGLE_USERNAME=<username>
export KAGGLE_KEY=<kaggle key>
pip install kaggle
kaggle competitions download -c imaterialist-fashion-2020-fgvc7
unzip imaterialist-fashion-2020-fgvc7 -d Data
# setting up jupyter-notebook, refer to https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52
jupyter notebook --generate-config
echo -e "c = get_config()\nc.NotebookApp.ip = '*'\nc.NotebookApp.open_browser = False\nc.NotebookApp.port = 8888\n" >> ~/.jupyter/jupyter_notebook_config.py
# setting up git
git config --global core.editor "vim"
git config --global user.name <username>
git config --global user.email <email>
echo -e "[alias]\n\tst = status\n\tbr = branch\n\tco = checkout\n\tlg = log --decorate --oneline -10" >> ~/.gitconfig
cd Code
mkdir Log
mkdir Model
nohup jupyter notebook --allow-root > jupyter_notebook.log &

# TPU
following this guide for the instances setup - https://cloud.google.com/tpu/docs/creating-deleting-tpus#console_1
make sure the instance is set up with - Under Identity and API access > Access scopes, select Allow full access to all Cloud APIs.
A tutorial from GCP https://cloud.google.com/tpu/docs/tutorials/resnet-pytorch, but we opened the TPU using the GCP GUI
cd code
conda activate /anaconda3/envs/torch-xla-1.6
export TPU_NAME=imat  # MUST BE THE SAME NAME AS THE PROJECT NAME!
export TPU_IP_ADDRESS=10.109.175.50  # REPLACE WITH INTERNAL IP OF TPU FROM GCP
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"  # DO NOT REPLACE THIS WITH THE ACTUAL IP
# to disable torch.jit which does not support many of the XLA methods, also run this:
export PYTORCH_JIT=0

Might also need to install the following packages:
sudo apt install libavcodec-dev
sudo apt install libavformat-dev
sudo apt install libswscale-dev
```

# COCO

**Download COCO dataset**

https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9

```
project
└─── Code
│        git clone...
└─── Data
     └─── annotations
     │        json files...
     └─── train2017
     │        train images...
     └─── val2017
              validation images...
```

**CUDA**
1. uninstall any previous cuda and nvidia driver - https://askubuntu.com/questions/206283/how-can-i-uninstall-a-nvidia-driver-completely
2. reboot system
3. Install using the instructions here (runfile) - https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal

**Install requirements**
```
cat requirements.txt | xargs -n 1 -L 1 pip3 install
```

**Train a new model**

```
PYTORCH_JIT=0 python train.py --load-model False --batch-size 5 --num-epochs 30 --num-workers 1
```


# memory leak

```
Memory usage 9 [66.3]
Filename: /mnt/383231D23231963A/TAU MSc/Semester 4/Thesis/Project/Code/coco_eval.py

Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
    34   8356.2 MiB   8356.2 MiB           1       @profile
    35                                             def update(self, predictions):
    36   8356.2 MiB      0.0 MiB           1           img_ids = list(np.unique(list(predictions.keys())))
    37   8356.2 MiB      0.0 MiB           1           self.img_ids.extend(img_ids)
    38                                         
    39   8407.9 MiB      0.0 MiB           3           for iou_type in self.iou_types:
    40   8407.4 MiB     50.3 MiB           2               results = self.prepare(predictions, iou_type)
    41   8407.7 MiB      0.5 MiB           2               coco_dt = loadRes(self.coco_gt, results) if results else COCO()
    42   8407.7 MiB      0.0 MiB           2               coco_eval = self.coco_eval[iou_type]
    43                                         
    44   8407.7 MiB      0.0 MiB           2               coco_eval.cocoDt = coco_dt
    45   8407.7 MiB      0.0 MiB           2               coco_eval.params.imgIds = list(img_ids)
    46   8407.9 MiB      0.9 MiB           2               img_ids, eval_imgs = evaluate(coco_eval)
    47                                         
    48   8407.9 MiB      0.0 MiB           2               self.eval_imgs[iou_type].append(eval_imgs)


Memory usage 9 [66.7]
Filename: /mnt/383231D23231963A/TAU MSc/Semester 4/Thesis/Project/Code/coco_eval.py

Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
    34   8408.2 MiB   8408.2 MiB           1       @profile
    35                                             def update(self, predictions):
    36   8408.2 MiB      0.0 MiB           1           img_ids = list(np.unique(list(predictions.keys())))
    37   8408.2 MiB      0.0 MiB           1           self.img_ids.extend(img_ids)
    38                                         
    39   8484.9 MiB      0.0 MiB           3           for iou_type in self.iou_types:
    40   8484.1 MiB     75.2 MiB           2               results = self.prepare(predictions, iou_type)
    41   8484.4 MiB      0.5 MiB           2               coco_dt = loadRes(self.coco_gt, results) if results else COCO()
    42   8484.4 MiB      0.0 MiB           2               coco_eval = self.coco_eval[iou_type]
    43                                         
    44   8484.4 MiB      0.0 MiB           2               coco_eval.cocoDt = coco_dt
    45   8484.4 MiB      0.0 MiB           2               coco_eval.params.imgIds = list(img_ids)
    46   8484.9 MiB      1.0 MiB           2               img_ids, eval_imgs = evaluate(coco_eval)
    47                                         
    48   8484.9 MiB      0.0 MiB           2               self.eval_imgs[iou_type].append(eval_imgs)

```
