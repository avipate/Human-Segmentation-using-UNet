# Importing required libraries
import tensorflow as tf
import numpy as np
import os
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage import color
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

IMG_HEIGHT = 256
IMG_WIDTH = 256
CHANNELS = 3

training_images_names = os.listdir('data/Training_Images/')
training_masks_names = os.listdir('data/masks/')

x = np.zeros((len(training_images_names), IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype='unit8')
y = np.zeros((len(training_masks_names), IMG_HEIGHT, IMG_WIDTH, 1))
