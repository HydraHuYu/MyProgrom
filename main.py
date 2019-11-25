from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, BatchNormalization, \
    MaxPooling1D, Activation, Flatten, CuDNNLSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import np_utils, plot_model
from keras.backend.tensorflow_backend import set_session
from sklearn import preprocessing
import init_data, os, pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# 训练参数
batch_size = 256
epochs = 2000
BatchNorm = True  # 是否批量归一化

train_x, train_y, test_x, test_y = init_data.init_data(length=2048, path=r'1750', count1=660,count2=25)
init_data.save_csv(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, path=r'data1750')
train_x, train_y, test_x, test_y = init_data.load_csv(path=r'data1750')
train_x, test_x = init_data.min_max_scaler(train_x=train_x, test_x=test_x)
train_y, test_y, num_class = init_data.one_hot(train_y=train_y, test_y=test_y)
train_x, test_x = train_x[:, :, np.newaxis], test_x[:, :, np.newaxis]
# 输入数据的维度
input_shape = train_x.shape[1:]

print('训练样本维度:', train_x.shape)
print('训练lable维度:', train_y.shape)
print('测试样本的维度', test_x.shape)
print('测试lable维度:', test_y.shape)
# 实例化序贯模型

model = Sequential()
# 搭建输入层，第一层卷积,指定input_shape，
model.add(Conv1D(filters=16, kernel_size=64, strides=16,
                 padding='same', kernel_regularizer=l2(1e-4),input_shape=input_shape))
if BatchNorm:
    model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2))

# 第二层卷积
model.add(Conv1D(filters=32, kernel_size=3, strides=1,
                 padding='same', kernel_regularizer=l2(1e-4)))
if BatchNorm:
    model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))

# 第三层卷积
model.add(Conv1D(filters=64, kernel_size=3, strides=1,
                 padding='same', kernel_regularizer=l2(1e-4)))
if BatchNorm:
    model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
# 第四层卷积
model.add(Conv1D(filters=64, kernel_size=3, strides=1,
                 padding='same', kernel_regularizer=l2(1e-4)))
if BatchNorm:
    model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
# 第五层卷积
model.add(Conv1D(filters=64, kernel_size=3, strides=1,
                 padding='valid', kernel_regularizer=l2(1e-4)))
if BatchNorm:
    model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))

# 从卷积到全连接需要展平
model.add(Flatten())

# 添加全连接层
model.add(Dense(units=100, activation='relu', kernel_regularizer=l2(1e-4)))
model.add(Activation('relu'))
# 增加输出层
model.add(Dense(units=num_class, activation='softmax', kernel_regularizer=l2(1e-4)))

# 编译模型 评价函数和损失函数相似，不过评价函数的结果不会用于训练过程中
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# 开始模型训练
history = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs,
                    verbose=0, validation_data=(test_x,test_y)	, shuffle=True,
                    callbacks=[init_data.ActivationCallback()])

score = model.evaluate(x=test_x, y=test_y, verbose=0)
print('测试集上的维度:', test_x.shape,'测试集上的lable维度:', test_y.shape)
print("测试集上的loss：", score[0],"测试集上的acc:",score[1])
init_data.Plt(history=history.history, path=r'res')
plot_model(model=model, to_file='./res/diagnosis.png', show_shapes=True)
