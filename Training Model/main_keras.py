# import lib
import os                           # duong dan
import cv2                          # computer vision
import tensorflow as tf             # 
import pytesseract as pt            # nhan dien text
import plotly.express as px 
import matplotlib.pyplot as plt 
import numpy as np                  # xu li du lieu
import pandas as pd                 # xu li du lieu
import xml.etree.ElementTree as xet # xu li du lieu
from glob import glob               # xu li du lieu
from skimage import io              # doc hinh anh
from shutil import copy
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#-----------------------------------------------------------------------------

# parsing data from .xml
if not os.path.exists('./DataSet/images'):
    print("Folder images is not exist!")
    exit(0)
path = glob('./DataSet/images/*.xml') # Lấy toàn bộ định dạng .xml
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
    filepath_image = './DataSet/images/' + filename_image
    return filepath_image
image_path = list(df['filepath'].apply(getFilename))

#-----------------------------------------------------------------------------

#data processing
labels = df.iloc[ : , 1 : ].values # lay du lieu
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    # process data
    h,w,d = img_arr.shape
    load_image = load_img(image,target_size=(224,224)) # Thay doi kich thuoc anh
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 # normalize
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax)
    data.append(norm_load_image_arr)
    output.append(label_norm)
    
    # verify data for each image
    fig = px.imshow(img_arr)
    fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title= image + ' with bounding box')
    fig.add_shape(type='rect',x0=xmin, x1=xmax, y0=ymin, y1=ymax, xref='x', yref='y',line_color='red')
    outfile = './verify/output' + str(ind) + '.html'
    print('Verifying image', image, 'in', outfile)
    fig.write_html(outfile)
        
X = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0) # ham chia random du lieu de train va test voi ti le 8:2

# base model
inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3))) #Kich thuoc dau vao 224*224 va co 3 kenh mau (RGB)

# head model
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel) # chuyển đổi 1 tensor nhiều chiều thành một vector một chiều

# Sử dụng 2 lớp Dense với 500 và 250 neuron đều sử dụng hàm kích hoạt ReLU vào mô hình.
# Relu( Rectified Linear Unit) là một hàm kích hoạt phổ biến trong các mạng thần kinh. Hàm này = max(0, input_value).
headmodel = Dense(500,activation="relu")(headmodel) 
headmodel = Dense(250,activation="relu")(headmodel)
# Lớp Dense cuối với 4 neuron, sử dụng hàm sigmoid. Số lượng neuron tương ứng với số lớp cần phân loại
headmodel = Dense(4,activation='sigmoid')(headmodel) # Hàm kích hoạt sigmoid sẽ đưa ra các giá trị xác suất nằm trong khoảng từ 0 đến 1

# dinh nghia model
model = Model(inputs=inception_resnet.input,outputs=headmodel)
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)) # loss function tính theo mean squared error (mse)

# model training and save
tfb = TensorBoard('object_detection') # Theo dõi quá trình huấn luyện 
history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=100,validation_data=(x_test,y_test),callbacks=[tfb])
model.save('./my_model.keras')


# load model
if os.path.exists('./my_model.keras'):
    model = tf.keras.models.load_model('./my_model.keras')
    print('Model loaded Sucessfully')
else:
    print('Model is not found!')
