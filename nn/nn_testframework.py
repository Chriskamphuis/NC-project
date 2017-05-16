import numpy as np

from cnn_1c2fc import *
#from solver import *

data = np.zeros((1,3,6,7))
data[:,0,:,:] = np.ones((6,7))

print(data)

model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=100, reg=0.001)

pred, cache = model.predict(data)

print(pred)

loss, grads = model.train(1, pred, cache)

print(loss)

print(grads)

"""
solver = Solver(model, data,
                num_epochs=1, batch_size=1,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()
"""
