import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.models import Sequential


class Population():

    def __init__(self, name='genetic player', input_size=(3, 6, 7),
                 crossover=0.01, mutation=0.1,
                 population_size=5, elitism=True):
        # Parameters
        self.name = name
        self.crossover = crossover
        self.mutation = mutation
        self.population_size = population_size
        self.elitsm = elitism

        # Network
        self.input_size = input_size
        self.networks = [self.build_network(input_size) for _
                         in range(population_size)]

    def build_network(self, input_size):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu',
                  input_shape=self.input_size))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def predict(self, board):
        return [n.predict(board) for n in self.networks]

    def copy(self, network):
        weights = network.get_weights()
        new_network = self.build_network(self.input_size)
        new_network.set_weights(weights)
        return new_network

    def train(self, fitness_list):
        fscores = np.array([float(i)/sum(fitness_list) for i in fitness_list])
        new_pop = []
        if self.elitsm:
            new_pop = [self.networks[np.argmax(fscores)]]
            for net in np.random.choice(self.networks,
                                        size=self.population_size-1,
                                        p=fscores):
                new_pop.append(self.copy(net))
        else:
            for net in np.random.choice(self.networks,
                                        size=self.population_size,
                                        p=fscores):
                new_pop.append(self.copy(net))
        self.networks = new_pop

    # Swaps conv filters using crossover with probability self.crossover
    def apply_crossover(self):
        amount_filters = 0
        for network in p.networks:
            for layer in network.layers:
                if type(layer) is Conv2D:
                    weights = layer.get_weights()[0]
                    amount_filters += weights.shape[2] * weights.shape[3]
        to_change = np.random.choice(amount_filters,
                                     size=(int(self.crossover *
                                               amount_filters *
                                               2)),
                                     replace=False)
        fil_from = to_change[:len(to_change)/2]
        fil_to = to_change[len(to_change)/2:]

        # TODO: check if fil_from and fil_to are equal length
        for i in range(len(fil_from)):
            self.swap_filters(fil_from[i], fil_to[i])

    def swap_filters(self, filter1, filter2):
        # TODO: Swap the filters at locations filter1 and filter2
        return

    def apply_mutation(self):
        for i in range(len(self.networks)):
            if i == 0 and self.elitsm:
                continue
            else:
                for layer in self.networks[i].layers:
                    if type(layer) == Conv2D or type(layer) == Dense:
                        weights = layer.get_weights()
                        w = weights[0]
                        b = weights[1]
                        w += np.random.normal(scale=self.mutation, size=w.shape)
                        b += np.random.normal(scale=self.mutation, size=b.shape)
                        layer.set_weights([w, b])

if __name__ == '__main__':
    p = Population()
    p.apply_crossover()
