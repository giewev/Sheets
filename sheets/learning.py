import numpy as np
import os
import random
from scipy.io import wavfile

def softmax(w):
    e = np.exp(np.clip(w, -500, 500))
    dist = e / np.sum(e)
    return dist

def softmax_entropy_derivative(o, t):
    return o - t

def sigmoid_square_derivative(o, t):
    return (t - o) * sigmoid_derivative(o)

def relu(v):
    return v.clip(min=0)

def relu_derivative(o):
    d = np.ones(o.shape)
    d[np.where(o < 0)] = 0

    return d

def sigmoid(x):
    return 1.0 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

class Activation(object):
    def __init__(self, forward, backward, backward_loss):
        self.forward = forward
        self.backward = backward
        self.backward_loss = backward_loss

relu_activation = Activation(relu, relu_derivative, None)
sigmoid_activation = Activation(sigmoid, sigmoid_derivative, None)
softmax_entropy_activation = Activation(softmax, None, softmax_entropy_derivative)
sigmoid_square_activation = Activation(sigmoid, None, sigmoid_square_derivative)

class NeuralNetwork(object):
    def __init__(self, shape, activations = None, rate = 0.1):
        self.weights = []
        self.biases = []
        self.learning_rate = rate

        for x in zip(shape[:-1], shape[1:]):
            self.weights.append(np.random.randn(x[0], x[1]))
            self.biases.append(np.random.randn(1, x[1]))

        if activations == None:
            activations = [relu_activation] * (len(self.weights) - 1)
            activations.append(softmax_entropy_activation)

        self.activations = activations

    def calculate(self, inputs):
        if len(inputs.shape) == 2 and inputs[0] == 1:
            j = inputs
        else:
            j = inputs.reshape((1, -1))

        signal = inputs
        for weight, bias, activate in zip(self.weights, self.biases, self.activations):
            signal = activate(np.dot(signal[-1], weight) + bias)
        return signal

    def get_backprop_deltas(self, inputs, targets):
        if len(inputs.shape) == 2 and inputs[0] == 1:
            signals = [inputs]
        else:
            signals = [inputs.reshape((1, -1))]

        for weight, bias, activate in zip(self.weights, self.biases, self.activations):
            signals.append(activate(np.dot(signals[-1], weight) + bias))

        d_output = self.activations[-1].backward_loss(signals[-1], targets)
        weight_deltas = [self.learning_rate * np.dot(signals[-1].T, d_output)]
        bias_deltas = [self.learning_rate * d_output]

        for w_index in range(1, len(self.weights)):
            d_output = np.dot(d_output, self.weights[-w_index].T)
            d_output *= self.activations[-w_index].backward(signals[-w_index])
            weight_deltas.append(self.learning_rate * np.dot(signals[-w_index - 1].T, d_output))
            bias_deltas.append(self.learning_rate * d_output)
        return (list(reversed(weight_deltas)), list(reversed(bias_deltas)))

    def apply_deltas(self, deltas):
        for x in range(len(self.weights)):
            self.weights[x] -= deltas[0][x]
            self.biases[x] -= deltas[1][x]

    def train_backprop(self, inputs, targets):
        self.apply_deltas(self.get_backprop_deltas(inputs, targets))

    def train_backprop_batch(self, input_list, target_list):
        deltas = None
        for x in zip(input_list, target_list):
            if deltas == None:
                deltas = self.get_backprop_deltas(x[0], x[1])
            else:
                new_deltas = self.get_backprop_deltas(x[0], x[1])
                for y in range(len(self.weights)):
                    deltas[0][y] += new_deltas[0][y]
                    deltas[1][y] += new_deltas[1][y]
        self.apply_deltas(deltas)

class LogisticNetwork(object):
    def __init__(self, shape, rate = 0.1):
        self.weights = []
        self.biases = []
        self.learning_rate = rate

        for x in zip(shape[:-1], shape[1:]):
            self.weights.append(np.random.randn(x[0], x[1]))
            self.biases.append(np.random.randn(1, x[1]))

    def calculate(self, inputs):
        signal = inputs
        for x in zip(self.weights, self.biases):
            signal = sigmoid(np.dot(signal[-1], x[0]) + x[1])
        return signal

    def train(self, inputs, targets):
        signal = [inputs]
        for x in zip(self.weights, self.biases):
            signal.append(sigmoid(np.dot(signal[-1], x[0]) + x[1]))

        errors = []
        errors.append(sigmoid_derivative(signal[-1]) * (targets - signal[-1]))
        for x in reversed(signal[1:-1]):
            errors.append(errors[-1].sum() * sigmoid_derivative(x))
        errors = reversed(errors)

class ReluNetwork(object):
    def __init__(self, shape, rate = 0.1):
        self.weights = []
        self.biases = []
        for x in zip(shape[:-1], shape[1:]):
            self.weights.append(np.random.randn(x[0], x[1]))
            self.biases.append(np.random.randn(1, x[1]))

        self.learning_rate = rate
 
    def calculate(self, inputs):
        if len(inputs.shape) == 2 and inputs[0] == 1:
            j = inputs
        else:
            j = inputs.reshape((1, -1))

        for w,b in zip(self.weights[:-1], self.biases[:-1]):
            j = relu(np.dot(j, w) + b)

        return softmax(np.dot(j, self.weights[-1]) + self.biases[-1])

    def get_backprop_deltas(self, inputs, targets):
        if len(inputs.shape) == 2 and inputs[0] == 1:
            signals = [inputs]
        else:
            signals = [inputs.reshape((1, -1))]

        for w,b in zip(self.weights[:-1], self.biases[:-1]):
            signals.append(relu(np.dot(signals[-1], w) + b))

        output = softmax(np.dot(signals[-1], self.weights[-1]) + self.biases[-1])

        d_output = output - targets
        weight_deltas = [self.learning_rate * np.dot(signals[-1].T, d_output)]
        bias_deltas = [self.learning_rate * d_output]

        for w_index in range(1, len(self.weights)):
            d_output = np.dot(d_output, self.weights[-w_index].T)
            d_output[np.where(signals[-w_index] < 0)] = 0
            weight_deltas.append(self.learning_rate * np.dot(signals[-w_index - 1].T, d_output))
            bias_deltas.append(self.learning_rate * d_output)
        return (list(reversed(weight_deltas)), list(reversed(bias_deltas)))

    def apply_deltas(self, deltas):
        for x in range(len(self.weights)):
            self.weights[x] -= deltas[0][x]
            self.biases[x] -= deltas[1][x]

    def train_backprop(self, inputs, targets):
        self.apply_deltas(self.get_backprop_deltas(inputs, targets))

    def train_backprop_batch(self, input_list, target_list):
        deltas = None
        for x in zip(input_list, target_list):
            if deltas == None:
                deltas = self.get_backprop_deltas(x[0], x[1])
            else:
                new_deltas = self.get_backprop_deltas(x[0], x[1])
                for y in range(len(self.weights)):
                    deltas[0][y] += new_deltas[0][y]
                    deltas[1][y] += new_deltas[1][y]
        self.apply_deltas(deltas)

class SoundClassifier(object):
    def __init__(self, data_set, frame_width = 1000, frame_jump = 100, test_ratio = 0.2):
        self.test_ratio = test_ratio
        self.load_data(data_set, frame_width, frame_jump)
        self.standardize_data()

        self.model = ReluNetwork((frame_width, int(frame_width / 5), self.training_buckets.shape[0]), rate = 0.0001)

    def train(self, batch_size = 1):
        input_list = []
        target_list = []
        for x in range(batch_size):
            class_index = random.randrange(0, self.training_buckets.shape[0])
            target_array = np.zeros(self.training_buckets.shape[0])
            target_array[class_index] = 1
            example_data = self.training_buckets[class_index][random.randrange(0, self.training_buckets[class_index].shape[0])]
            input_list.append(example_data)
            target_list.append(target_array)

        self.model.train_backprop_batch(input_list, target_list)

    def accuracy_ratio(self):
        correct = 0
        incorrect = 0
        for target_class in range(self.test_buckets.shape[0]):
            for example_data in self.test_buckets[target_class]:
                if np.argmax(self.model.calculate(example_data)) == target_class:
                    correct += 1
                else:
                    incorrect += 1
        return correct / float(correct + incorrect)

    def standardize_data(self):
        count = 0
        total = 0
        for x in self.training_buckets:
            total += x.sum()
            count += x.size
        mean = total / float(count)

        variance = 0
        for x in self.training_buckets:
            variance += ((x - mean) ** 2).sum()
        variance /= float(count)

        self.training_buckets -= mean
        self.training_buckets /= variance ** 0.5
        self.test_buckets -= mean
        self.test_buckets /= variance ** 0.5

    def load_data(self, data_set, frame_width, frame_jump):
        self.classes = dict()
        self.training_buckets = []
        self.test_buckets = []
        for x in os.listdir(data_set):
            if not os.path.isdir(data_set + '/' + x):
                continue

            self.classes[x] = len(self.classes)
            new_data = []
            for y in os.listdir(data_set + '/' + x):
                new_data.extend(SoundClassifier.load_examples(data_set + '/' + x + '/' + y, frame_width, frame_jump))
            
            random.shuffle(new_data)
            test_data_count = int(len(new_data) * self.test_ratio)
            self.training_buckets.append(np.array(new_data[test_data_count:]))
            self.test_buckets.append(np.array(new_data[:test_data_count]))
        self.training_buckets = np.array(self.training_buckets)
        self.test_buckets = np.array(self.test_buckets)

    @staticmethod
    def load_examples(path, frame_width, frame_jump):
        examples = []
        sample_rate, waveform = wavfile.read(path)
        for channel_index in range(waveform.shape[1]):
            channel = waveform[:, channel_index]
            index = 0
            while index + frame_width < len(channel):
                examples.append(channel[index : index + frame_width])
                index += frame_jump
        return examples