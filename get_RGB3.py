import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

#path 为trian图片的路径
path="D:/Code/python/Data/train"

#print(os.listdir(path))  #检验路径是否存在


#是否进行最大训练次数
RUN_MAX = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
BATCH_SIZE=15

def Get_data(path):
    filenames = os.listdir(path)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    return df

#获取数据，将图片地址存储在df(DataFrame中)
df = Get_data(path)

#設計卷積訓練模型
model = Sequential()

#选择3，3的卷积核进行卷积，使用2，2池化窗口进行池化,每一层卷积进行0.25的dropout
#建模型(卷积—relu-池化-卷积-relu-池化-卷积—relu-池化-全连接)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization()) #用于提高训练的速度，对训练数据集进行归一化的操作，即将原始数据减去其均值后，再除以其方差 。
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#压平，即把多维的输入一维化，从卷积层到全连接层的过渡
model.add(Flatten())

#全连接层的实现
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#打印网络函数
model.summary()

# 提前停止训练的callbacks
earlystop = EarlyStopping(patience=10)

#检测val_acc，评价指标不在提升时，减少学习率（每次以0.5的比例进行减少），学习率的下限为0.00001
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=2,  #触发的循环次数
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

#回归函数
callbacks = [earlystop, learning_rate_reduction]

#由于要使用ImageDataGenerator需要将类型变为字符串，所以进行替换
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

#分割数据集
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True) #还原索引
validate_df = validate_df.reset_index(drop=True)

#数据量
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]


#设置图片生成器，并进行数据增强
train_datagen = ImageDataGenerator(
    rotation_range=15,  #随机转动的角度
    rescale=1./255, #重放缩
    shear_range=0.1, #剪切强度
    zoom_range=0.2, #随机缩放的幅度
    horizontal_flip=True, #进行随机水平翻转
    width_shift_range=0.1,#图片水平偏移的幅度
    height_shift_range=0.1 #图片竖直偏移的幅度
)

#批量录入训练数据
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    path,
    x_col='filename',  #图片地址
    y_col='category', #标签
    target_size=IMAGE_SIZE,
    class_mode='categorical', #2D one-hot 编码标签
    batch_size=BATCH_SIZE #批量数据的尺寸
)

#同上，设置验证集
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE
)


epochs=3 if RUN_MAX else 50
validation_steps = total_validate//BATCH_SIZE  #进行整数除法
steps_per_epoch=total_train//BATCH_SIZE

history = model.fit_generator(
    train_generator,
    epochs=epochs, #迭代次数
    validation_data=validation_generator,
    validation_steps=validation_steps,
    steps_per_epoch=steps_per_epoch, #每次迭代训练的数据量
    callbacks=callbacks
)


model.save('model.h5')

