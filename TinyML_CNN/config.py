import random

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging
import os


"""File contains configurations for datasets, models, and optimization techniques
to be used throughout the repository."""

# Use the dotenv library to load the dataset and project location
# from a .env file
dotenv.load_dotenv()
# Setting parameters for plotting
plt.rcParams["figure.figsize"] = (15.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

# Set the logging to tensorflow warning 
logging.getLogger("tensorflow").setLevel(logging.WARNING)

# Fashion MNIST classes for classifier testing
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Dataset configuration for image sizes and 
# number of classes for classification tasks
class datacfg:
    h, w = 64, 64
    img_size = (64, 64)
    num_classes = len(classes)
    classes = classes
    in_channels = 1

# Model architecture configuration (not currently set )
class modelcfg:
    ...

# Configuration for model training 
class traincfg:
    epochs = 10
    es_patience = 7

    ssim_loss_weight = 0.85
    l1_loss_weight = 0.1
    edge_loss_weight = 0.9

# Configuration to be used across the entire
# repository
class metacfg:
    # do_overfit = True
    do_overfit = False
    do_shuffle = True
    do_subsample = True
    batch_size = 8
    take_first_n = 30

    is_cluster = os.path.exists("/cluster")

    save_model_dir = ""
    save_cfiles_dir = ""
    save_test_data_dir = ""
    base_dataset_dir = ""

    tmpdir = os.getenv("TMPDIR")
    logdir = ""
    path_to_project_dir = os.environ["path_to_project_dir"]
    ckpt_dir = f"{path_to_project_dir}/code/ckpt"

# Placeholder config (Don't know what this is for)
class cfg(datacfg, modelcfg, traincfg, metacfg):
    ...

# Set the directories for saving H5, tflite, and flat buffer model
# files to disk
metacfg.save_model_dir = f"{metacfg.path_to_project_dir}/code/models"
metacfg.save_cfiles_dir = f"{metacfg.path_to_project_dir}/code/cfiles"
metacfg.save_test_data_dir = f"{metacfg.path_to_project_dir}/code/test_data"

if metacfg.is_cluster:
    metacfg.base_dataset_dir = os.path.join(
        metacfg.tmpdir, os.environ["base_dataset_dir"]
    )
else:
    metacfg.base_dataset_dir = os.environ["base_dataset_dir"]
    metacfg.logdir = "/tmp"