import numpy as np


# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义反向传播函数
def backpropagation(inputs, targets, num_hidden):
    # 初始化权重
    input_size = inputs.shape[1]
    output_size = targets.shape[1]
    hidden_size = num_hidden

    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    weights_hidden_output = np.random.randn(hidden_size, output_size)

    # 设置学习率和迭代次数
    learning_rate = 0.1
    num_iterations = 1000

    for iteration in range(num_iterations):
        # 前向传播
        hidden_activations = np.dot(inputs, weights_input_hidden)
        hidden_outputs = sigmoid(hidden_activations)

        output_activations = np.dot(hidden_outputs, weights_hidden_output)
        output_outputs = sigmoid(output_activations)

        # 计算误差
        output_errors = targets - output_outputs
        hidden_errors = np.dot(output_errors, weights_hidden_output.T)

        # 反向传播更新权重
        weights_hidden_output += learning_rate * np.dot(hidden_outputs.T,
                                                        output_errors * output_outputs * (1 - output_outputs))
        weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_errors * hidden_outputs * (1 - hidden_outputs))

    return weights_input_hidden, weights_hidden_output


# 示例用法
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

weights_input_hidden, weights_hidden_output = backpropagation(inputs, targets, num_hidden=4)
print("Weights (input to hidden):\n", weights_input_hidden)
print("Weights (hidden to output):\n", weights_hidden_output)
