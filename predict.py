from __future__ import division, print_function
import cv2
import os
import numpy as np
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
import keras

from keras.models import load_model

app = Flask(__name__)

#IMAGE_WIDTH=112
#IMAGE_HEIGHT=112
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)


#img_path = 'C:/Users\赵震洋\PycharmProjects\catdog/uploads'
img_path = 'D:/catdog/uploads'
model_path = 'D:/catdog/model.h5'
#model_path = 'D:/catdog/cats_dogs_CNN_1.h5'

#model = load_model(model_path)

def load_a_model():
    global model

    keras.backend.clear_session()
    model = load_model(model_path)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

    img_path = 'D:/catdog/uploads'

    load_a_model()

    img_path = os.path.join(img_path, f.filename)
    print(f.filename)
    f.save(img_path)
    print(img_path)

    img_path_data = cv2.imread((img_path))
    img_data = cv2.resize(img_path_data, IMAGE_SIZE)  # 重新设置图片尺寸
    img_data = (img_data / 255)

    one_data =np.array(img_data)
    one_data = one_data.reshape((1, (one_data.shape)[0], (one_data.shape)[1], 3))
    print('---------------')

    #model = load_model(model_path)
    predicted_one = model.predict(one_data)
    print(predicted_one)
    print(model_path)

    try:
        predicted_one = np.argmax(predicted_one, axis=-1)

        result_cat = '这是一只猫!'
        result_dog = '这是一只狗!'
    except:
        return '223'
    if predicted_one== 0:

        return result_cat
    else:

        return result_dog
    return None



if __name__ == '__main__':
    app.run()




