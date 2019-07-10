from keras.models import load_model
import numpy as np
import os
import cv2

model_path = "D:/Code/python/Data/model/train/RGB/epoch50/model.h5"
img_path = "D:/Code/python/Data/test1/dog.3.jpg"

IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

def Get_one_data(path):
    image_data = cv2.imread((path))
    image_data = cv2.resize(image_data, IMAGE_SIZE)  # 重新设置图片尺寸
    image_data =(image_data / 255)
    return image_data

model = load_model(model_path)
print(model_path)
one_data = Get_one_data(img_path)
# print(one_data)
# print(one_data.shape)
# one_data = np.array(one_data)
# one_data=one_data.reshape((1,(one_data.shape)[0],(one_data.shape)[1],1))  #修改(102, 64, 64)形状为 (102, 64, 64, 1)最后一维是图片厚度
# predicted_one=model.predict(one_data)
# predicted_one=np.round(predicted_one,decimals=2)
# predicted_one=[1 if value>0.5 else 0 for value in predicted_one]
# print(predicted_one)


print(one_data)
print(one_data.shape)
one_data = np.array(one_data)
one_data=one_data.reshape((1,(one_data.shape)[0],(one_data.shape)[1],3))  #修改(102, 64, 64)形状为 (102, 64, 64, 1)最后一维是图片厚度
predicted_one=model.predict(one_data)
predicted_one= np.argmax(predicted_one, axis=-1)

print(predicted_one)


