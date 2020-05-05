import numpy as np
import scipy.special
import time
import pandas as pd
import argparse
import os

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=1, type=int, help="batch size")
args = parser.parse_args()

# 创建神经网络类
class neuralNetwork(object):
    # 初始化
    def __init__(self, input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate):
        """
        参数说明：
        input_nodes:输入层节点数
        hidden1_nodes:隐藏层1节点数
        hidden2_nodes:隐藏层2节点数
        output_ndes:输出层节点数
        learning_rate:学习率
        """
        self.inodes = input_nodes
        self.h1_nodes = hidden1_nodes
        self.h2_nodes = hidden2_nodes
        self.onodes = output_nodes

        self.lr = learning_rate

        # 激活函数
        self.activate_function = lambda x: scipy.special.expit(x)

        # 初始化权值
        self.wih_1 = np.random.normal(0.0, pow(self.h1_nodes, -0.5), (self.h1_nodes, self.inodes))

        # 第一个隐藏层与第二个隐藏层之间的权重
        self.wh1_h2 = np.random.normal(0.0, pow(self.h2_nodes, -0.5), (self.h2_nodes, self.h1_nodes))

        # 隐藏层与输出层的权重
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.h2_nodes))

        pass

    # 训练函数
    def train(self, input_lists, target_lists):
        # 将输入数据转换成列矩阵
        inputs = np.array(input_lists, ndmin=2).T
        # 将目标数据转换成列矩阵
        targets = np.array(target_lists, ndmin=2).T

        # 计算隐藏层1输入
        hidden1_inputs = np.dot(self.wih_1, inputs)
        # 计算隐藏层1输出
        hidden1_outputs = self.activate_function(hidden1_inputs)


        # 计算隐藏层2输入
        hidden2_inputs = np.dot(self.wh1_h2, hidden1_outputs)
        # 计算隐藏层2输出
        hidden2_outputs = self.activate_function(hidden2_inputs)


        # 计算输出层输入
        output_inputs = np.dot(self.who, hidden2_outputs)
        # 计算输出层输出
        output_outputs = self.activate_function(output_inputs)

        """
        反向传播
        """
        # 计算误差
        errors = targets - output_outputs

        # 隐藏层2误差
        hidden2_errors = np.dot(self.who.T, errors)
        # print(hidden2_errors.shape)
        # 隐藏层1误差
        hidden1_errors = np.dot(self.wh1_h2.T, hidden2_errors)
        # print(hidden1_errors.shape)
        # 输入层误差
        input_errors = np.dot(self.wih_1.T, hidden1_errors)
        # print(input_errors.shape)
        # print(hidden1_outputs.shape)

        # 更新权重
        self.who += self.lr * np.dot((errors * output_outputs * (1 - output_outputs)), np.transpose(hidden2_outputs))

        self.wh1_h2 += self.lr * np.dot((hidden2_errors * hidden2_outputs * (1 - hidden2_outputs)), np.transpose(hidden1_outputs))

        self.wih_1 += self.lr * np.dot((hidden1_errors * hidden1_outputs * (1 - hidden1_outputs)), np.transpose(inputs))

    # 前向传播
    def query(self, input_lists):
        # 将输入数据转换成列数据
        inputs = np.array(input_lists, ndmin=2).T

        # 计算隐藏层1输入
        hidden1_inputs = np.dot(self.wih_1, inputs)
        # 计算隐藏层1输出
        hidden1_outputs = self.activate_function(hidden1_inputs)

        # 计算隐藏层2输入
        hidden2_inputs = np.dot(self.wh1_h2, hidden1_outputs)
        # 计算隐藏层2输出
        hidden2_outputs = self.activate_function(hidden2_inputs)

        # 计算输出层输入
        output_inputs = np.dot(self.who, hidden2_outputs)
        # 计算输出层输出
        output_outputs = self.activate_function(output_inputs)

        return output_outputs


def loadData(path, train=True):
    # 加载数据
    if train:
        training_data = pd.read_csv(path + 'train.csv')
        #train = training_data.drop(['label'], axis=1)
        training_data = np.array(training_data)

        # 将训练集的标签取出
        labels = []
        for i in training_data:
            labels.append(i[0])

        return training_data, labels
    else:
        # 将MNIST测试集转换成列表
        test = pd.read_csv(path + 'test.csv')
        test = np.array(test)

        return test

def saveData(result, path, name='mnist_NN.csv'):
    saveDir = '../output'
    # 如果要保存的文件夹没有，就创建
    if not os.path.exists(saveDir):
        os.makedir(saveDir)

    sample = pd.read_csv(path + 'sample_submission.csv')

    temp = pd.DataFrame({'ImageId':sample['ImageId'],
                        'Label':result})

    temp.to_csv(saveDir + '/' + name, index=False)


if __name__ == '__main__':

    # 定义一些参数训练MNIST所使用的节点数
    INPUT_NODES = 784
    HIDDEN1_NODES = 200
    HIDDEN2_NODES = 300
    OUT_DIM = 10
    EPOCHS = args.EPOCHS

    # 学习率
    LR = 0.1

    # 创建神经网络实例
    n = neuralNetwork(INPUT_NODES, HIDDEN1_NODES, HIDDEN2_NODES, OUT_DIM, LR)
    path = '../data/'

    # 加载数据
    train, labels = loadData(path)
    test = loadData(path, False)

    TRAIN_LENGTH = len(labels)

    # 训练神经网络
    start = time.time()
    for e in range(EPOCHS):
        print('EPOCH {} / {}'.format(e+1, EPOCHS))
        for i, record in enumerate(train):
            # 将数据分割
            # print(record)

            # 处理数据
            inputs = (np.asfarray(record[1:], dtype=int) / 255.0 * 0.99) + 0.01

            # 分离出目标值
            targets = np.zeros(OUT_DIM) + 0.01

            targets[int(record[0])] = 0.99
            n.train(inputs, targets)

            if i % 1000 == 0:
                print('RUNNING... {:.2f} %'.format(i/TRAIN_LENGTH * 100))

    finish = time.time()
    print('Time Cost: ', finish-start)


    # 在测试集上测试
    result = []
    for record in test:
        # 分离标签和像素数据

        inputs = (np.asfarray(record, dtype=int) / 255.0 * 0.99) + 0.01

        # 前向传播
        outputs = n.query(inputs)

        label = np.argmax(outputs)

        result.append(label)


    saveData(result, path)
