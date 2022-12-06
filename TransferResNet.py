import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import splitfolders
import shutil

# or import splitfolders
test_dir = "input/test"

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                           shuffle=True,
                                                           batch_size=BATCH_SIZE,
                                                           image_size=IMG_SIZE)

IMG_SHAPE = IMG_SIZE + (3,)

model50 = load_model("model/SatNet50.h5")
model101 = load_model("model/SatNet101.h5")

loss, accuracy = model50.evaluate(test_dataset)
print('Test accuracy :', accuracy)

loss, accuracy = model101.evaluate(test_dataset)
print('Test accuracy :', accuracy)

