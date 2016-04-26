import numpy as np
import os
import random
from scipy.io import wavfile

def soft_max(w):
    e = np.exp(np.clip(w, -500, 500))
    dist = e / np.sum(e)
    return dist

def relu(v):
    return v.clip(min=0)

def vector_sigmoid(x):
    return 1.0 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

class DataPoint(object):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

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
            signal = vector_sigmoid(np.dot(signal[-1], x[0]) + x[1])
        return signal

    def train(self, inputs, targets):
        signal = [inputs]
        for x in zip(self.weights, self.biases):
            signal.append(vector_sigmoid(np.dot(signal[-1], x[0]) + x[1]))

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

        return soft_max(np.dot(j, self.weights[-1]) + self.biases[-1])

    def get_backprop_deltas(self, inputs, targets):
        if len(inputs.shape) == 2 and inputs[0] == 1:
            signals = [inputs]
        else:
            signals = [inputs.reshape((1, -1))]

        for w,b in zip(self.weights[:-1], self.biases[:-1]):
            signals.append(relu(np.dot(signals[-1], w) + b))

        output = soft_max(np.dot(signals[-1], self.weights[-1]) + self.biases[-1])

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
        self.confidence_threshold = 0.75
        self.test_ratio = test_ratio
        self.load_data(data_set, frame_width, frame_jump)
        self.standardize_data()

        print(len(self.classes))
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