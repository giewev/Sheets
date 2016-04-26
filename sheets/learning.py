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

class DataPoint(object):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

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
            signal = inputs
        else:
            signal = inputs.reshape((1, -1))

        for weight, bias, activate in zip(self.weights, self.biases, self.activations):
            signal = activate.forward(np.dot(signal, weight) + bias)
        return signal

    def get_backprop_deltas(self, inputs, targets):
        if len(inputs.shape) == 2 and inputs[0] == 1:
            signals = [inputs]
        else:
            signals = [inputs.reshape((1, -1))]

        for weight, bias, activate in zip(self.weights, self.biases, self.activations):
            signals.append(activate.forward(np.dot(signals[-1], weight) + bias))

        d_output = self.activations[-1].backward_loss(signals[-1], targets)
        weight_deltas = [self.learning_rate * np.dot(signals[-2].T, d_output)]
        bias_deltas = [self.learning_rate * d_output]

        for w_index in range(1, len(self.weights)):
            d_output = np.dot(d_output, self.weights[-w_index].T)
            d_output *= self.activations[-w_index - 1].backward(signals[-w_index - 1])
            weight_deltas.append(self.learning_rate * np.dot(signals[-w_index - 2].T, d_output))
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

class LogisticNetwork(NeuralNetwork):
    def __init__(self, shape, rate = 0.1):
        activations = [sigmoid_activation] * (len(shape) - 2)
        activations.append(sigmoid_square_activation)

        super(LogisticNetwork, self).__init__(shape, activations, rate)

class ReluNetwork(NeuralNetwork):
    def __init__(self, shape, rate = 0.1):
        activations = [relu_activation] * (len(shape) - 2)
        activations.append(softmax_entropy_activation)
        
        super(ReluNetwork, self).__init__(shape, activations, rate)

class SoundClassifier(object):
    def __init__(self, data_set, frame_width = 1000, frame_jump = 100, test_ratio = 0.2):
        self.confidence_threshold = 0.75
        self.test_ratio = test_ratio
        self.load_data(data_set, frame_width, frame_jump)
        self.standardize_data()

        self.model = ReluNetwork((frame_width, int(frame_width / 5), len(self.classes)), rate = 0.0001)

    def train(self, batch_size = 1):
        input_list = []
        target_list = []
        for x in range(batch_size):
            example_data = random.choice(self.training_data)
            input_list.append(example_data.inputs)
            target_list.append(example_data.targets)

        self.model.train_backprop_batch(input_list, target_list)

    def accuracy_ratio(self):
        correct = 0
        incorrect = 0
        for example_data in self.test_data:
            output = self.threshold_vector(self.model.calculate(example_data.inputs))
            if np.all(output == example_data.targets):
                correct += 1
            else:
                incorrect += 1
        return correct / float(correct + incorrect)

    def standardize_data(self):
        count = 0
        total = 0
        for x in self.training_data:
            total += x.inputs.sum()
            count += x.inputs.size
        mean = total / float(count)

        variance = 0
        for x in self.training_data:
            variance += ((x.inputs - mean) ** 2).sum()
        variance /= float(count)

        self.training_data = np.array([DataPoint((x.inputs - mean) / (variance ** 0.5), x.targets) for x in self.training_data])
        self.test_data = np.array([DataPoint((x.inputs - mean) / (variance ** 0.5), x.targets) for x in self.test_data])

    def load_data(self, data_set, frame_width, frame_jump):
        self.load_classes(data_set)
        self.training_data = []
        self.test_data = []
        for x in os.listdir(data_set):
            if ".wav" in x.lower():
                new_data = SoundClassifier.load_examples(data_set + "/" + x, frame_width, frame_jump)
                new_data = self.build_data_points(new_data, x)
                random.shuffle(new_data)
                test_data_count = int(len(new_data) * self.test_ratio)
                self.training_data.extend(np.array(new_data[test_data_count:]))
                self.test_data.extend(np.array(new_data[:test_data_count]))
        self.training_data = np.array(self.training_data)
        self.test_data = np.array(self.test_data)

    def build_data_points(self, data_inputs, file_name):
        target = np.zeros(len(self.classes))
        for x in file_name.split("-")[:-1]:
            target[self.classes[x]] = 1
        return [DataPoint(x, target) for x in data_inputs]

    def load_classes(self, data_set):
        self.classes = dict()
        f = open(data_set + "/Classes.txt")
        for x in f:
            self.classes[x.replace("\n", "")] = len(self.classes)

    def threshold_vector(self, v):
        v[np.where(v >= self.confidence_threshold)] = 1
        v[np.where(v < self.confidence_threshold)] = 0
        return v

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