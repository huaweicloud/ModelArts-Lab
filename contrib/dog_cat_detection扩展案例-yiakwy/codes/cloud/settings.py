import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))

# dataset
DATA_DIR = os.path.abspath(os.path.join(ROOT, "../../", "data"))

# output
OUTPUT_DIR = os.path.abspath(os.path.join(ROOT, "../../", "output"))

# pretrained model
CAT_DOG_PRETRAINED_MODEL = os.path.join(DATA_DIR, "pretrained", "dog_cat_dataset/weights.best.checkpoint.hdf5")

# used for input data
HEIGHT = 128
WIDTH = 128
CHANNEL = 3

# TRAINING
EPOCHS=30
BATCH_SIZE=32

# classes
NUM_CLASSES=10

