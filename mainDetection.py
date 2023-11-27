import pandas as pd
import numpy as np


def load_data(path_to_file):
    return pd.read_csv(path_to_file).to_numpy(dtype=np.float64)


all_data = load_data("files/creditcard_2023.csv")


def split_data_X_Y(data):
    initial_dimensions = data.shape
    all_X_data = data[:, 1:initial_dimensions[1] - 1]
    all_Y_data = data[:, initial_dimensions[1] - 1:initial_dimensions[1]]
    return all_X_data, all_Y_data


X_data, Y_data = split_data_X_Y(all_data)


def normalize_data(X):
    m = X.shape[1]
    miu = 1. / m * np.sum(X, axis=0)
    X -= miu
    sigma_patrat = 1. / m * np.sum(X ** 2, axis=0)
    sigma = np.sqrt(sigma_patrat)
    X /= sigma
    return X


X_data_norm = normalize_data(X_data)


def split_and_shuffle_data(data_X, data_Y):
    p = np.random.permutation(len(data_Y))
    data_X = data_X[p]
    data_Y = data_Y[p]
    length = data_X.shape[0]
    train_size = int(np.floor(0.95 * length))  # 95% of data goes to training
    dev_size = int(np.floor(0.025 * length))  # 2.5% of data goes to dev and test set
    train_X_data = data_X[0:train_size, :]  # am pus 70 si 15, 15
    train_Y_data = data_Y[0:train_size, :]
    dev_X_data = data_X[train_size:train_size + dev_size, :]
    dev_Y_data = data_Y[train_size:train_size + dev_size, :]
    test_X_data = data_X[train_size + dev_size:, :]
    test_Y_data = data_Y[train_size + dev_size:, :]
    return train_X_data, train_Y_data, dev_X_data, dev_Y_data, test_X_data, test_Y_data


train_X, train_Y, dev_X, dev_Y, test_X, test_Y = split_and_shuffle_data(X_data_norm, Y_data)

# make the transpose for each of them
train_X = train_X.T
train_Y = train_Y.T
dev_X = dev_X.T
dev_Y = dev_Y.T
test_X = test_X.T
test_Y = test_Y.T


def create_mini_batch(X, Y, mini_batch_size=128):
    m = X.shape[1]
    mini_batches = []
    num_complete_minibatches = m // mini_batch_size
    for k in range(num_complete_minibatches):
        mini_batch_X = X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    mini_batch_X = X[:, mini_batch_size * num_complete_minibatches:]
    mini_batch_Y = Y[:, mini_batch_size * num_complete_minibatches:]
    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)
    return mini_batches


train_batches = create_mini_batch(train_X, train_Y)


def initialize_parameters(nr_units, nr_input):
    w = np.random.randn(nr_units, nr_input) * np.sqrt(2 / nr_input)
    b = np.zeros((nr_units, 1))
    return w, b


def initialize_all_parameters(input_dims, list_of_units_per_layer: list):
    parameters = {}
    w_tmp, b_tmp = initialize_parameters(list_of_units_per_layer[0], input_dims)
    parameters[f"W{1}"] = w_tmp
    parameters[f"b{1}"] = b_tmp
    for i in range(1, len(list_of_units_per_layer)):
        w_tmp, b_tmp = initialize_parameters(list_of_units_per_layer[i], list_of_units_per_layer[i - 1])
        parameters[f"W{i + 1}"] = w_tmp
        parameters[f"b{i + 1}"] = b_tmp
    return parameters


def sigmoid_activation_function(Z):
    return np.exp(Z) / (1 + np.exp(Z))


def relu_activaton_function(Z):
    return np.maximum(0, Z)


def cost_function(Y, Y_hat):
    sigmoid_sum = - np.multiply(Y, np.log(Y_hat)) - np.multiply((1 - Y), np.log(1 - Y_hat))
    sigmoid_cost = np.sum(sigmoid_sum) / Y.shape[1]
    return sigmoid_cost


def forward_propagation(X, parameters, activation_list):
    cache = {"A0": X}
    for i in range(1, len(activation_list) + 1):
        cache[f"Z{i}"] = np.dot(parameters[f"W{i}"], cache[f"A{i - 1}"]) + parameters[f"b{i}"]
        if activation_list[i - 1] == "relu":
            cache[f"A{i}"] = relu_activaton_function(cache[f"Z{i}"])
        elif activation_list[i - 1] == "sigmoid":
            cache[f"A{i}"] = sigmoid_activation_function(cache[f"Z{i}"])
        else:
            print("eroare")
    return cache


def cost_function(Y, Yhat, epsilon=10e-7):
    m = Y.shape[1]
    costul = -np.sum(np.multiply(Y, np.log(Yhat + epsilon)) + np.multiply(1 - Y, np.log(1 - Yhat + epsilon)))
    costul /= m
    # costul = costul.squeeze()
    return costul


def backward_propagation(X, Y, parameters, cache, list_of_activations):
    m = X.shape[1]
    grads = {}
    dZ = cache[f"A{len(list_of_activations)}"] - Y
    grads[f"dW{len(list_of_activations)}"] = 1 / m * np.dot(dZ, cache[f"A{len(list_of_activations) - 1}"].T)
    grads[f"db{len(list_of_activations)}"] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA = np.dot(parameters[f"W{len(list_of_activations)}"].T, dZ)

    for i in reversed(range(1, len(list_of_activations))):
        dZ = np.multiply(dA, np.int64(cache[f"A{i}"] > 0))
        grads[f"dW{i}"] = 1 / m * np.dot(dZ, cache[f"A{i - 1}"].T)
        grads[f"db{i}"] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(parameters[f"W{i}"].T, dZ)
        # print(i)
    return grads


def gradient_descend(parameters: dict, grads: dict, learning_rate):
    for val in parameters.keys():
        parameters[val] = parameters[val] - np.dot(learning_rate, grads[f"d{val}"])
    return parameters


def init_adam(parameters: dict):
    V = {}
    S = {}
    for el in parameters.keys():
        S[el] = np.zeros(parameters[el].shape)
        V[el] = np.zeros(parameters[el].shape)
    return V, S


def adam_optimization(parameters: dict, grads: dict, V: dict, S: dict, iteration_number, learning_rate, beta1=0.9,
                      beta2=0.999, epsilon=10e-7):
    for param in parameters.keys():
        V[param] = beta1 * V[param] + (1 - beta1) * grads[f"d{param}"]  # momentum
        S[param] = beta2 * S[param] + (1 - beta2) * (grads[f"d{param}"] ** 2)  # RMSprop
        Vcorr = V[param] / (1 - beta1 ** iteration_number)
        Scorr = S[param] / (1 - beta2 ** iteration_number)
        parameters[param] = parameters[param] - learning_rate * Vcorr / (np.sqrt(Scorr) + epsilon)
    return parameters, V, S


def model(mini_batches, iterations=100, learning_rate=0.01):
    list_activations = ["relu", "relu", "relu", "relu", "sigmoid"]
    list_units = [30, 64, 32, 16, 1]
    nr_batches = len(mini_batches)
    afis = nr_batches // 20
    all_costs = []
    parameters = initialize_all_parameters(29, list_units)
    for i in range(iterations):
        costul = 0
        print(f"Epoch {i} ", end="")
        for i in range(len(mini_batches)):
            X, Y = mini_batches[i]
            cache = forward_propagation(X, parameters, list_activations)
            costul += cost_function(Y, cache[f"A{len(list_activations)}"])
            grads = backward_propagation(X, Y, parameters, cache, list_activations)
            parameters = gradient_descend(parameters, grads, learning_rate)
            if i % afis == 0:
                print("=", end="")
        all_costs.append(costul / afis)
        if i % 1 == 0:
            print(f">: {costul / afis}")
    return parameters, all_costs


def calculate_accuracy(parameters_trained: dict, list_of_activations, X_input, Y_output):
    #     m = Y_output.shape[1]
    cache = forward_propagation(X_input, parameters_trained, list_of_activations)
    Y_hat = cache[f"A{len(list_of_activations)}"]
    corecte = 0
    for i in range(Y_hat.shape[1]):
        if Y_hat[0][i] >= 0.5 and Y_output[0][i] == 1:
            corecte += 1
        if Y_hat[0][i] < 0.5 and Y_output[0][i] == 0:
            corecte += 1
    return 100 * corecte / Y_output.shape[1]


def model_with_adam(mini_batches, dev_x, dev_y,iterations=10, learning_rate=0.01, beta1=0.9, beta2=0.999):
    list_activations = ["relu", "relu", "relu", "relu", "sigmoid"]
    list_units = [30, 64, 32, 16, 1]
    nr_batches = len(mini_batches)
    afis = nr_batches // 20
    parameters = initialize_all_parameters(29, list_units)
    V, S = init_adam(parameters)
    all_costs = []
    for i in range(iterations):
        costul = 0
        print(f"Epoch {i} ", end="")
        for j in range(len(mini_batches)):
            X, Y = mini_batches[j]
            cache = forward_propagation(X, parameters, list_activations)
            costul += cost_function(Y, cache[f"A{len(list_activations)}"])
            grads = backward_propagation(X, Y, parameters, cache, list_activations)
            parameters, V, S = adam_optimization(parameters, grads, V, S, i + 1, learning_rate, beta1, beta2)
            if j % afis == 0:
                print("=", end="")
        all_costs.append(costul / afis)
        print(f">: {costul / afis}\t acuratetea:", end="")
        print(calculate_accuracy(parameters, list_activations, dev_x, dev_y))
    return parameters, all_costs


final_params, costul_arr = model_with_adam(train_batches, dev_X, dev_Y,20, 0.01)

print(calculate_accuracy(final_params, ["relu", "relu", "relu", "relu", "sigmoid"], train_X, train_Y))
