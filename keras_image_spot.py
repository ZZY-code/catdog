# coding: utf8
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, normalization
from keras.optimizers import RMSprop


TRAIN_PATH = r'D:\\Pycharm\\PycharmProject\\Project_0\\train\\'  # 训练集路径
TEST_PATH = r'D:\\Pycharm\\PycharmProject\\Project_0\\test\\'  # 测试集路径
SAVE_PATH = r'D:\\Pycharm\\PycharmProject\\Project_0\\save\\cats_dogs_CNN_1.h5'  # 模型保存路径+保存文件名
IMAGE_SIZE = 112  # 图像预处理大小
BATCH_SIZE = 30  # 每批次图像数量


# 训练及测试图像预处理
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,  # 随机旋转的度数范围
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # 剪切强度
    zoom_range=0.2,  # 随机缩放范围
    horizontal_flip=True,  # 随机水平翻转
    fill_mode='nearest',
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_PATH,
    classes=['dogs','cats'],
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
test_generator = test_datagen.flow_from_directory(
    directory=TEST_PATH,
    classes=['dogs','cats'],
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# CNN 模型构建
model = Sequential()
# 卷积层1，输出(IMAGE_SIZE, IMAGE_SIZE, 32)
model.add(Convolution2D(
    batch_input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_last',
    # data_format默认为'channels_last',对应输入尺寸为 (batch, height, width, channels)
))
model.add(normalization.BatchNormalization())  # 批标准化
model.add(Activation('relu'))

# 池化层1，输出(56, 56, 32)
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

# 卷积层2，输出(56, 56, 64)
model.add(Convolution2D(64, 5, strides=1, padding='same'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))

# 池化层2，输出(28, 28, 64)
model.add(MaxPooling2D(2, 2, 'same'))

# 卷积层3，输出(28, 28, 128)
model.add(Convolution2D(128, 5, strides=1, padding='same'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))

# 池化层3，输出(14, 14, 128)
model.add(MaxPooling2D(2, 2, 'same'))

# 卷积层4，输出(14, 14, 256)
model.add(Convolution2D(256, 5, strides=1, padding='same'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))

# 池化层4，输出(7, 7, 256)
model.add(MaxPooling2D(2, 2, 'same'))
model.add(Dropout(0.25))  # 随机丢弃四分之一节点，防止过拟合

# 全连接层1，输入(256 * 7 * 7) = (12544),输出(2048)
model.add(Flatten())  # 展平张量为1D
model.add(Dense(2048))
model.add(Activation('relu'))

# 全连接层2，输出(512)
model.add(Dense(512))
model.add(Activation('relu'))

# 全连接层3，输出两种类别
model.add(Dense(2))
model.add(Activation('sigmoid'))

# 输出神经网络特征图的维度变化情况
# model.summary()

# 配置训练模型
model.compile(
    optimizer=RMSprop(lr=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 开始训练
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=800,
    epochs=50,
    verbose=2,  # 日志显示模式:每轮一次
    validation_data=test_generator,
    validation_steps=25
)
'''
# 保存模型
model.save(SAVE_PATH)

import matplotlib.pyplot as plt
# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''