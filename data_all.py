from keras.models import load_model
import numpy as np
import os
import cv2
from keras.utils import to_categorical


IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)


def Get_train_data(path):
    print(os.listdir(path))
    counter = 0
    i = 0
    label = []
    data = []
    for file in os.listdir(path):
        # cv2.imread()返回(height, width, channels)    cv2.IMREAD_GRAYSCALE 灰色图像
        # os.path.join() 合并目录路径和文件名，生成文件路径
        image_data = cv2.imread(os.path.join(path, file))
        image_data = cv2.resize(image_data, IMAGE_SIZE)  # 重新设置图片尺寸
        if file.startswith("cat"):
            label.append(0)
        elif file.startswith("dog"):
            label.append(1)
        try:
            data.append(image_data / 255)
        except:
            label = label[:len(label) - 1]
        counter += 1
        if counter % 1000 == 0:
            print(counter, " image data retreived")
        i = i + 1

    print(i)  # 查看数据是否全部录入
    return data,label




img_path = "D:/Code/python/Data/test5"

#model = load_model('model.h5')
model = load_model('D:/Code/python/Data/model/train/RGB/epoch50/model.h5')
print("D:/Code/python/Data/model/train/RGB/epoch50/model.h5")

test_data,test_label = Get_train_data(img_path)

# print(one_data)
# print(one_data.shape)
test_data = np.array(test_data)
test_data=test_data.reshape(((test_data.shape)[0],(test_data.shape)[1],(test_data.shape)[2],3))  #修改(102, 64, 64)形状为 (102, 64, 64, 1)最后一维是图片厚度
test_label=np.array(test_label)
test_label = to_categorical(test_label)

test_loss, test_acc = model.evaluate(test_data, test_label)

print("test_loss",test_loss)
print("test_acc",test_acc)

