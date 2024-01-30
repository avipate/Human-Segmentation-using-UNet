# Importing required libraries
import tensorflow as tf
import numpy as np
import os
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 256
IMG_WIDTH = 256
CHANNELS = 3

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_images_names = os.listdir('data/Training_Images/')
training_masks_names = os.listdir('data/masks/')

data_generator = ImageDataGenerator(rescale=1./255)  # Rescale to normalize pixel values

X = np.zeros((len(training_images_names), IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype='uint8')
y = np.zeros((len(training_masks_names), IMG_HEIGHT, IMG_WIDTH, 1))

input_directory = 'data/Training_Images/'
output_directory = 'data/resized_training/'

generator = data_generator.flow_from_directory(
    input_directory,
    target_size=(256, 256),  # Resize images to (256, 256)
    batch_size=32,
    class_mode='categorical',  # Set class mode according to your data
    save_to_dir=output_directory,  # Save the resized images to the output directory
    save_format='png'  # Save images in PNG format, change as needed
)

# def resize_images(input_dir, output_dir, size=(256, 256)):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     for filename in os.listdir(input_dir):
#         input_path = os.path.join(input_dir, filename)
#         output_path = os.path.join(output_dir, filename)
#
#         # Open the image using Pillow
#         img = Image.open(input_path)
#
#         # Resize the image
#         img = img.resize(size)
#
#         # Convert the image to a NumPy array
#         img_array = np.array(img)
#
#         # Save the resized image
#         output_img = Image.fromarray(img_array)
#         output_img.save(output_path)
#
#
# input_directory = 'data/Training_Images'
# output_directory = 'data/resized_training'
# resize_images(input_directory, output_directory)
