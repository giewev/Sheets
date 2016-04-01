import numpy as np
import os
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
            signal = vector_sigmoid(np.dot(signal, x[0]) + x[1])
        return signal

    def train(self, inputs, targets):
        signal = [inputs]
        for x in zip(self.weights, self.biases):
            signal.append(vector_sigmoid(np.dot(signal, x[0]) + x[1]))

        errors = []
        errors.append(self.activation_derivative(signal[-1]) * (targets - signal[-1]))
        for x in reversed(signal[1:-1]):
            errors.append(errors[-1].sum() * self.activation_derivative(x))
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

    def train(self, inputs, targets):
        if len(inputs.shape) == 2 and inputs[0] == 1:
            signals = [inputs]
        else:
            signals = [inputs.reshape((1, -1))]

        for w,b in zip(self.weights[:-1], self.biases[:-1]):
            signals.append(relu(np.dot(signals[-1], w) + b))

        output = soft_max(np.dot(signals[-1], self.weights[-1]) + self.biases[-1])

        d_output = output - targets
        self.weights[-1] -= self.learning_rate * np.dot(signals[-1].T, d_output)
        self.biases[-1]  -= self.learning_rate * d_output

        for w_index in range(1, len(self.weights)):
            d_output = np.dot(d_output, self.weights[-w_index].T)
            d_output[np.where(signals[-w_index])] = 0
            self.weights[-w_index - 1] -= self.learning_rate * np.dot(signals[-w_index - 1].T, d_output)
            self.biases[-w_index - 1]  -= self.learning_rate * d_output

class SoundClassifier(object):
    def __init__(self, data_set, frame_width = 3200, frame_jump = 100):
        self.load_data(data_set, frame_width, frame_jump)
        self.standardize_data()

        self.model = LogisticNetwork((frame_width, int(frame_width / 4), self.data_buckets.shape[0]))

    def standardize_data(self):
        count = 0
        total = 0
        for x in self.data_buckets:
            total += x.sum()
            count += x.size
        mean = total / float(count)

        variance = 0
        for x in self.data_buckets:
            variance += ((x - mean) ** 2).sum()
        variance /= float(count)

        self.data_buckets -= mean
        self.data_buckets /= variance ** 0.5

    def load_data(self, data_set, frame_width, frame_jump):
        self.classes = dict()
        self.data_buckets = []
        for x in os.listdir(data_set):
            if not os.path.isdir(data_set + '/' + x):
                continue

            self.classes[x] = len(self.classes)
            new_data = []
            for y in os.listdir(data_set + '/' + x):
                new_data.extend(SoundClassifier.load_examples(data_set + '/' + x + '/' + y, frame_width, frame_jump))
            self.data_buckets.append(np.array(new_data))
        self.data_buckets = np.array(self.data_buckets)

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