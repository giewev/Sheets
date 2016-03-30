from sheets.learning import ReluNetwork

import numpy as np

net = ReluNetwork((7,2,3,3,4))
# print(net.calculate(np.random.randn(7)))
net.train(np.random.randn(7),np.random.randn(4))

from scipy.io import wavfile

fs, data = wavfile.read('notes/a3/a3-1.wav')
print(type(fs))
print(type(data))
print(fs)
print(data)
print(data.shape)