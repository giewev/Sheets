from sheets.learning import ReluNetwork

import numpy as np

net = ReluNetwork((7,2,3,3,4))
print(net.calculate(np.random.randn(7)))
net.train(np.random.randn(7),np.random.randn(4))