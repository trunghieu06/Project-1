# import lib
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet
from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ----------------------------------------------------------------------

# load model
model = tf.keras.models.load_model('./my_model.keras')
print('Model loaded Sucessfully')

def object_detection(path, filename):
    # Read image
    image = load_img(path)
    image = np.array(image, dtype=np.uint8)
    image1 = load_img(path, target_size=(224, 224))
    # Data preprocessing
    # Convert into array and get the normalized output
    image_arr_224 = img_to_array(image1)/255.0
    h, w, d = image.shape
    test_arr = image_arr_224.reshape(1, 224, 224, 3)
    # Make predictions
    coords = model.predict(test_arr)
    # Denormalize the values
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    # Draw bounding on top the image
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    print(pt1, pt2)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    # Convert into bgr
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename), image_bgr)
    return coords


def save_text(filename, text):
    name, ext = os.path.splitext(filename)
    with open('./static/text/{}.txt'.format(name), mode='w') as f:
        f.write(text)

def OCR(path, filename):
    img = np.array(load_img(path))
    cods = object_detection(path, filename)
    xmin, xmax, ymin, ymax = cods[0]
    roi = img[ymin:ymax, xmin:xmax]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    magic_color = apply_brightness_contrast(gray, brightness=40, contrast=70)
    cv2.imwrite('./static/roi/{}'.format(filename), roi_bgr)
    # extract text from cropped image
    text = pt.image_to_string(magic_color, lang='eng', config='--psm 6')
    print(text)
    save_text(filename, text)
    return text

# adjusts brightness and contrast
def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf