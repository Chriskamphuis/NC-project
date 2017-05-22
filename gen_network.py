import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.models import Sequential

input_s = (3, 6, 7)  # Zetten op board shape


def model():
    m = Sequential()
    m.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_s))
    m.add(Conv2D(32, (3, 3), activation='relu'))
    m.add(Flatten())
    m.add(Dense(100, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(1, activation='sigmoid'))
    return m

# Create 20 models as a population
population = [model() for _ in range(20)]
