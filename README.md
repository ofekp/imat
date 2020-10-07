# Introduction


# Article
Architecture - link to image

# Installation
pip install -r requirements.txt

# Creating a new H5PY dataset

While not a requirement, performing this step will greately improve training times.
If you wish to skip this step, remeber to use `--h5py-dataset false` when training.

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

# Visualization

Start jupyter note book
`nohup jupyter notebook --allow-root > jupyter_notebook.log`

Start `imat_visualization.ipynb` notebook

# TODO(ofekp): complete this using the article!