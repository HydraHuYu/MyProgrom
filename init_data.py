from scipy.io import loadmat
import os, pathlib
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import np_utils, plot_model
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


class ActivationCallback(Callback):
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}
        self.val_acc = 0
        self.val_loss = 100
        filepath_model = './log/model.h5'
        filepath_epoch = './log/epoch.txt'
        filepath_history = './log/history.txt'
        path_model = pathlib.Path(filepath_model)
        path_epoch = pathlib.Path(filepath_epoch)
        path_history = pathlib.Path(filepath_history)
        if path_model.exists() and path_epoch.exists() and path_history.exists():
            self.model = load_model(filepath_model)
            file = open(filepath_epoch, 'r+', encoding='utf-8')
            self.epoch = eval(file.read())
            file.close()
            file = open(filepath_history, 'r+', encoding='utf-8')
            self.history = eval(file.read())
            file.close()
            print('the last model, epoch and history all both are loaded.')
        else:
            print('The new model has been loaded')

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        if len(self.epoch):
            epoch = self.epoch[-1] + 1
        else:
            epoch = 1
        self.epoch.append(epoch)
        eps = 1e-8
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        print('epoch:%7d' % (self.epoch[-1]), ' - loss:%7f' % (logs['loss']), ' - acc:%7f' % (logs['acc']),
              ' - val_loss:%7f' % (logs['val_loss']), ' - val_acc:%7f' % (logs['val_acc']))

        if logs['val_acc'] > self.val_acc - eps and logs['val_loss']  < self.val_loss:
            self.val_acc = logs['val_acc']
            self.val_loss = logs['val_loss']
            filepath_best_model = './log/best_model.h5'
            self.model.save(filepath_best_model)
            print('epoch', epoch, 'the best model are soved！')
        if epoch % 100 == 0:
            filepath_model = './log/model.h5'
            filepath_epoch = './log/epoch.txt'
            filepath_history = './log/history.txt'
            path_model = pathlib.Path(filepath_model)
            path_epoch = pathlib.Path(filepath_epoch)
            path_history = pathlib.Path(filepath_history)
            self.model.save(filepath_model)
            file = open(filepath_epoch, 'w+', encoding='utf-8')
            file.write(str(self.epoch))
            file.close()
            file = open(path_history, 'w+', encoding='utf-8')
            file.write(str(self.history))
            file.close()
            Plt(self.history, path=r'log')
            print('epoch', epoch, 'model, epoch and history all both are soved.')


def Plt(history, path):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path + '/acc.png')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path + '/loss.png')
    plt.show()


def init_data(length, path, count1, count2):
    print(length, path, count1, count2)

    def capture(path):
        filenames = os.listdir(path)
        filenames.sort()
        files = {}
        num_class = 0
        for i in filenames:
            file_path = os.path.join(path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if path == r'1750' and 'X098' in key: continue
                if 'DE' in key:
                    files[num_class] = file[key].ravel()
                    num_class += 1
                    print(num_class - 1, file_path, key)
        return files, num_class

    def make_data(data, num_class, length, count1, count2):
        train_x = {}
        test_x = {}
        # 切割测试数据 无数据增强
        for i in range(num_class):
            test_data = data[i]
            test_sample = []
            for j in range(0, count2):
                sample = test_data[j * length:(j + 1) * length]
                test_sample.append(sample)
            test_x[i] = test_sample
            # 切割训练数据 无数据增强
            train_data = data[i]
            train_data = train_data[count2 * length:-1]
            length_train = len(train_data) - length
            data_enc = int(length_train / count1)
            print (i, length_train, data_enc)
            train_sample = []
            for j in range(0, count1):
                sample = train_data[j * data_enc:j * data_enc + length]
                train_sample.append(sample)
            train_x[i] = train_sample
        return train_x, test_x

    def add_labels(data, length, num_class):
        X = []
        Y = []
        label = 0
        for i in range(num_class):
            x = data[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        X = np.array(X).reshape(-1, length)
        Y = np.array(Y).reshape(-1, 1)
        return X, Y

    data, num_class = capture(path=path)
    train_x, test_x = make_data(data=data, length=length, num_class=num_class, count1=660, count2=25)
    train_x, train_y = add_labels(data=train_x, length=length, num_class=num_class)
    test_x, test_y = add_labels(data=test_x, length=length, num_class=num_class)
    return train_x, train_y, test_x, test_y


def save_csv(train_x, train_y, test_x, test_y, path):
    np.savetxt(path + '/train_data.csv', train_x, delimiter=',')
    np.savetxt(path + '/train_label.csv', train_y, delimiter=',', fmt='%d')
    np.savetxt(path + '/test_data.csv', test_x, delimiter=',')
    np.savetxt(path + '/test_label.csv', test_y, delimiter=',', fmt='%d')


def load_csv(path):
    train_x = np.loadtxt(path + '/train_data.csv', delimiter=',')
    train_y = np.loadtxt(path + '/train_label.csv', delimiter=',')
    test_x = np.loadtxt(path + '/test_data.csv', delimiter=',')
    test_y = np.loadtxt(path + '/test_label.csv', delimiter=',')
    return train_x, train_y, test_x, test_y

def min_max_scaler(train_x, test_x):
    data = np.vstack((train_x, test_x))
    scalar = preprocessing.MinMaxScaler().fit(data)
    train_x = scalar.transform(train_x)
    test_x = scalar.transform(test_x)
    return train_x, test_x


def one_hot(train_y, test_y):
    train_y = np.array(train_y).reshape([-1, 1])
    test_y = np.array(test_y).reshape([-1, 1])
    data = np.vstack((train_y, test_y))
    Encoder = preprocessing.OneHotEncoder()
    Encoder.fit(data)
    train_y = Encoder.transform(train_y).toarray()
    test_y = Encoder.transform(test_y).toarray()
    train_y = np.asarray(train_y, dtype=np.int32)
    test_y = np.asarray(test_y, dtype=np.int32)
    num_class = train_y.shape[1]
    return train_y, test_y, num_class


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = init_data(length=2048, path=r'1797', count1=660, count2=25)
    save_csv(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, path=r'data1797')
    train_x, train_y, test_x, test_y = load_csv(path=r'data1797')
    train_x, test_x = scalar_stand(train_x=train_x, test_x=test_x)
    train_y, test_y, num_class = one_hot(train_y=train_y, test_y=test_y)
    print('train_x维度', train_x.shape)
    print('train_y维度', train_y.shape)
    print('test_x维度', test_x.shape)
    print('test_y维度', test_y.shape)
