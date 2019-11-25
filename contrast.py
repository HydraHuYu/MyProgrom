from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, BatchNormalization, \
    MaxPooling1D, Activation, Flatten, CuDNNLSTM
from tensorflow.python.keras.models import Sequential, load_model
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
train_x1, train_y1, test_x1, test_y1 = init_data.init_data(length=2048, path=r'1772', count1=660,count2=25)
init_data.save_csv(train_x=train_x1, train_y=train_y1, test_x=test_x1, test_y=test_y1, path=r'data1772')
train_x1, train_y1, test_x1, test_y1 = init_data.load_csv(path=r'data1772')
train_y1, test_y1, num_class1 = init_data.one_hot(train_y=train_y1, test_y=test_y1)
train_x1, test_x1 = init_data.min_max_scaler(train_x=train_x1, test_x=test_x1)

train_x2, train_y2, test_x2, test_y2 = init_data.init_data(length=2048, path=r'1750', count1=660,count2=25)
init_data.save_csv(train_x=train_x2, train_y=train_y2, test_x=test_x2, test_y=test_y2, path=r'data1750')
train_x2, train_y2, test_x2, test_y2 = init_data.load_csv(path=r'data1750')
train_y2, test_y2, num_class2 = init_data.one_hot(train_y=train_y2, test_y=test_y2)
train_x2, test_x2 = init_data.min_max_scaler(train_x=train_x2, test_x=test_x2)

train_x3, train_y3, test_x3, test_y3 = init_data.init_data(length=2048, path=r'1730', count1=660,count2=25)
init_data.save_csv(train_x=train_x3, train_y=train_y3, test_x=test_x3, test_y=test_y3, path=r'data1730')
train_x3, train_y3, test_x3, test_y3 = init_data.load_csv(path=r'data1730')
train_y3, test_y3, num_class3 = init_data.one_hot(train_y=train_y3, test_y=test_y3)
train_x3, test_x3 = init_data.min_max_scaler(train_x=train_x3, test_x=test_x3)

test_x1, test_x2, test_x3 = test_x1[:, :, np.newaxis], test_x2[:, :, np.newaxis], test_x3[:, :, np.newaxis]

# 实例化序贯模型

model = load_model('./log/best_model.h5')
score = model.evaluate(x=test_x1, y=test_y1, verbose=0)
print("测试集1上的:loss：", score[0], "acc:", score[1])
predict = model.predict(test_x1)
for i in range(test_x1.shape[0]):
    sample = predict[i]
    temp = np.max(sample) * 100
    pos1 = sample.argmax()
    pos0 = test_y1[i].argmax()
    print(i, '%.2f%%' % temp, pos0, pos1,pos0 == pos1)

score = model.evaluate(x=test_x2, y=test_y2, verbose=0)
print("测试集2上的:loss：", score[0], "acc:", score[1])
predict = model.predict(test_x2)
for i in range(test_x2.shape[0]):
    sample = predict[i]
    temp = np.max(sample) * 100
    pos1 = sample.argmax()
    pos0 = test_y2[i].argmax()
    print(i, '%.2f%%' % temp, pos0, pos1,pos0 == pos1)

score = model.evaluate(x=test_x3, y=test_y3, verbose=0)
print("测试集2上的:loss：", score[0], "acc:", score[1])
predict = model.predict(test_x3)
for i in range(test_x3.shape[0]):
    sample = predict[i]
    temp = np.max(sample) * 100
    pos1 = sample.argmax()
    pos0 = test_y3[i].argmax()
    print(i, '%.2f%%' % temp, pos0, pos1,pos0 == pos1)