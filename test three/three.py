from csv import reader
from math import exp, log
from random import randrange, seed, random
import copy
import numpy as np
from matplotlib import pyplot as plt


# 读取csv文件和数据类型转换
def csv_loader(file):
    dataset = list()
    with open(file, 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# 字符串数据转换为浮点型
def str_to_float_converter(dataset):
    dataset = dataset[1:]
    for i in range(len(dataset[0]) - 1):
        for row in dataset:
            row[i] = float(row[i].strip())


# 观察数据转换为整型
def str_to_int_converter(dataset):
    dataset = dataset[1:]
    class_values = [row[-1] for row in dataset]
    unique_values = set(class_values)
    converter_dict = dict()
    for i, value in enumerate(unique_values):
        converter_dict[value] = i
    for row in dataset:
        row[-1] = converter_dict[row[-1]]


# 数据归一化
def normalization(dataset):
    for i in range(len(dataset[0]) - 1):
        col_values = [row[i] for row in dataset]
        max_value = max(col_values)
        min_value = min(col_values)
        for row in dataset:
            row[i] = (row[i] - min_value) / float(max_value - min_value)


# K折交叉验证拆分数据
def k_fold_cross_validation(dataset, n_folds):
    dataset_split = list()
    fold_size = int(len(dataset) / n_folds)
    dataset_copy = list(dataset)
    for i in range(n_folds):
        fold_data = list()
        while len(fold_data) < fold_size:
            index = randrange(len(dataset_copy))
            fold_data.append(dataset_copy.pop(index))
        dataset_split.append(fold_data)
    return dataset_split


# 计算准确性
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    accuracy = correct / float(len(actual)) * 100.0
    return accuracy


# 模型测试
def mode_scores(dataset, algo, n_folds, *args):
    dataset_split = k_fold_cross_validation(dataset, n_folds)
    scores = list()
    for fold in dataset_split:
        train = copy.deepcopy(dataset_split)
        train.remove(fold)
        train = sum(train, [])
        test = list()
        test = copy.deepcopy(fold)
        predicted = algo(train, test, *args)
        actual = [row[-1] for row in fold]
        accuracy = calculate_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


# 初始化神经网络
def initialize_network(n_inputs, n_hiddens, n_outputs):
    network = list()
    hidden_layer = [{'weight': [random() for i in range(n_inputs + 1)]} for i in range(n_hiddens)]
    network.append(hidden_layer)
    output_layer = [{'weight': [random() for i in range(n_hiddens + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# 计算激活值
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# 选用sigmoid作为激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 向前传递,获得输出结果
def forward_propagation(network, row):
    inputs = row
    for layer in network:
        inputs_new = list()
        for neuron in layer:
            activation = activate(neuron['weight'], inputs)
            if layer != network[-1]:
                neuron['output'] = sigmoid(activation)
                neuron['input'] = activation
                inputs_new.append(neuron['output'])
            else:
                neuron['output'] = activation
                neuron['input'] = activation
                inputs_new.append(neuron['output'])
        inputs = inputs_new
    return inputs


# 计算神经元输出值对输入值的导数
def transfer_derivative(input_):
    derivative = 1.0 / (1.0 + exp(-input_))
    return derivative


# 计算反向传播的误差
def back_propagation_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        if i - 1 >= 0:
            layer_pre = network[i - 1]
        else:
            layer_pre = None
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                neuron = layer[j]
                error = 0.0
                for neuron_latter in network[i + 1]:
                    error += (neuron_latter['weight'][j] * neuron_latter['delta'][j])
                neuron['error'] = error
                errors.append(error)

        else:
            for j in range(len(layer)):
                neuron = layer[j]
                error = neuron['output'] - expected[j]
                neuron['error'] = error
                errors.append(error)
        for j in range(len(layer)):
            if not layer_pre:
                continue
            else:
                neuron = layer[j]
                neuron['delta'] = list()
                for neuron_pre in layer_pre:
                    error_class = errors[j] * transfer_derivative(neuron_pre['input'])
                    neuron['delta'].append(error_class)


# 更新权重系数
def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weight'][j] -= learning_rate * neuron['error'] * inputs[j]
                neuron['weight'][-1] -= learning_rate * neuron['error']


# 训练神经网络
def train_network(network, train, learning_rate, n_epochs, n_outputs):
    for epoch in range(n_epochs):
        sum_error = 0.0
        for row in train:
            outputs = forward_propagation(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(outputs[i] - expected[i]) ** 2 for i in range(len(expected))])
            back_propagation_error(network, expected)
            update_weights(network, row, learning_rate)
        if epoch % 100 == 0:
            print('We are at epoch [%d] right now, The learning rate is [%.3f], the error is [%.3f]' % (
                epoch, learning_rate, sum_error))


# 预测
def make_prediction(network, row):
    outputs = forward_propagation(network, row)
    prediction = outputs.index(max(outputs))
    return prediction


# 反向传播
def back_propagation(train, test, learning_rate, n_epochs, n_hiddens):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set(row[-1] for row in train))
    network = initialize_network(n_inputs, n_hiddens, n_outputs)
    train_network(network, train, learning_rate, n_epochs, n_outputs)
    predictions = list()
    for row in test:
        prediction = make_prediction(network, row)
        predictions.append(prediction)
    return predictions


# # 定义反向传播函数
# def backpropagation(inputs, targets, num_hidden):
#     # 初始化权重
#     input_size = inputs.shape[1]
#     output_size = targets.shape[1]
#     hidden_size = num_hidden
#
#     np.random.seed(42)
#     weights_input_hidden = np.random.randn(input_size, hidden_size)
#     weights_hidden_output = np.random.randn(hidden_size, output_size)
#
#     # 设置学习率和迭代次数
#     learning_rate = 0.1
#     num_iterations = 1000
#
#     for iteration in range(num_iterations):
#         # 前向传播
#         hidden_activations = np.dot(inputs, weights_input_hidden)
#         hidden_outputs = sigmoid(hidden_activations)
#
#         output_activations = np.dot(hidden_outputs, weights_hidden_output)
#         output_outputs = sigmoid(output_activations)
#
#         # 计算误差
#         output_errors = targets - output_outputs
#         hidden_errors = np.dot(output_errors, weights_hidden_output.T)
#
# # 反向传播更新权重 weights_hidden_output += learning_rate * np.dot(hidden_outputs.T, output_errors * output_outputs * (1 -
# output_outputs)) weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_errors * hidden_outputs * (1 -
# hidden_outputs))
#
#     return weights_input_hidden, weights_hidden_output

# 17.开始测试
if __name__ == '__main__':
    file = 'diabetes.csv'
    dataset = csv_loader(file)
    str_to_float_converter(dataset)
    str_to_int_converter(dataset)
    dataset = dataset[1:]
    normalization(dataset)

    n_folds = 5
    learning_rate = 0.001
    n_epochs = 500
    n_hiddens = 10

    algo = back_propagation
    scores = mode_scores(dataset, algo, n_folds, learning_rate, n_epochs, n_hiddens)

    print('The scores of our model are : %s' % scores)
    print('The average score of our model is : %.3f%%' % (sum(scores) / float(len(scores))))
    X_test = np.linspace(0, 1, 100)
    x = [1, 2, 3, 4, 5]
    y = scores
    plt.plot(x, y)
    plt.show()
