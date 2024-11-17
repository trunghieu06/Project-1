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

# pip install tensorflow==2.12.0

from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#-----------------------------------------------------------------------------

# parsing data from .xml
if not os.path.exists('./DataSet/images'):
    print("Folder images is not exist!")
    exit(0)
path = glob('./DataSet/images/*.xml')
labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])
for i in path:
    info = xet.parse(i)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)
    labels_dict['filepath'].append(i)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

df = pd.DataFrame(labels_dict)
if os.path.exists('./DataSet/labels.csv'):
    os.remove('./DataSet/labels.csv')
df.to_csv('./DataSet/labels.csv',index=False)

def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    # filepath_image = os.path.join('./DataSet/images',filename_image)
    filepath_image = './DataSet/images/' + filename_image
    return filepath_image
image_path = list(df['filepath'].apply(getFilename))

# verify data
# file_path = image_path[0] # image N1.jpeg
# img = cv2.imread(file_path)
# # xmin-1093/ymin-645/xmax-1396/ymax-727
# fig = px.imshow(img)
# imm_title = image_path[0] + ' with bounding box'
# fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title= image_path[0] + ' with bounding box')
# fig.add_shape(type='rect',x0=1093, x1=1396, y0=645, y1=727, xref='x', yref='y',line_color='red')
# print('Verifying image', image_path[0], 'in file output.html')
# fig.write_html('./verify/output.html')

#-----------------------------------------------------------------------------

#data processing
labels = df.iloc[ : , 1 : ].values
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    # process data
    h,w,d = img_arr.shape
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax)
    data.append(norm_load_image_arr)
    output.append(label_norm)
    # verify data for each image
    # fig = px.imshow(img_arr)
    # fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title= image + ' with bounding box')
    # fig.add_shape(type='rect',x0=xmin, x1=xmax, y0=ymin, y1=ymax, xref='x', yref='y',line_color='red')
    # # print('Verifying image', image, 'in file output.html')
    # fig.write_html('./verify/output' + str(ind) + '.html')
        
X = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3)))
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500,activation="relu")(headmodel)
headmodel = Dense(250,activation="relu")(headmodel)
headmodel = Dense(4,activation='sigmoid')(headmodel)
model = Model(inputs=inception_resnet.input,outputs=headmodel)
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.summary()

# model training and save
# tfb = TensorBoard('object_detection')
# history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=100,validation_data=(x_test,y_test),callbacks=[tfb])
# model.save('./my_model.keras')


# load model
# if os.path.exists('./my_model.keras'):
#     model = tf.keras.models.load_model('./my_model.keras')
#     print('Model loaded Sucessfully')
# else:
#     print('Model is not found!')

# test model
# Create pipeline
# path = './DataSet/images/N1.jpeg'
# def object_detection(path):
#     image = load_img(path)
#     image = np.array(image,dtype=np.uint8)
#     image1 = load_img(path,target_size=(224,224))

#     # Data preprocessing
#     image_arr_224 = img_to_array(image1)/255.0
#     h,w,d = image.shape
#     test_arr = image_arr_224.reshape(1,224,224,3)

#     # Make predictions
#     coords = model.predict(test_arr)

#     # Denormalize the values
#     denorm = np.array([w,w,h,h])
#     coords = coords * denorm
#     coords = coords.astype(np.int32)

#     # Draw bounding on top the image
#     xmin, xmax,ymin,ymax = coords[0]
#     pt1 =(xmin,ymin)
#     pt2 =(xmax,ymax)
#     print(pt1, pt2)
#     cv2.rectangle(image,pt1,pt2,(0,255,0),3)
#     return image, coords

# image, cods = object_detection(path)

# fig = px.imshow(image)
# fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Image N1')
# fig.write_html('./test.html')
# print('Test image saved to test.html')

# img = np.array(load_img(path))
# xmin ,xmax,ymin,ymax = cods[0]
# roi = img[ymin:ymax,xmin:xmax]
# fig = px.imshow(roi)
# fig.update_layout(width=350, height=250, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Image N1 Cropped image')
# fig.write_html('./crop.html')
# print('Cropped image saved to crop.html')

# text = pt.image_to_string(roi)
# print('Image to string: ', text)