import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.models import Sequential
from tqdm import tqdm
import player
from game import Game
import matplotlib.pyplot as plt


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

    def predict(self, board, noise=True):
        prediction = [n.predict(board) for n in self.networks]
        if noise:
            prediction = [pred + np.random.normal(scale=0.01)
                          for pred in prediction]
        return prediction

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
        for network in self.networks:
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

        if len(fil_from) != len(fil_to):
            raise ValueError('fil from and fil to are not of equal length')

        for i in range(len(fil_from)):
            self.swap_filters(fil_from[i], fil_to[i])

    def swap_filters(self, filter1, filter2):
        if filter2 < filter1:
            tmp = filter1
            filter1 = filter2
            filter2 = tmp
        tmp = None
        i = 0
        n_t, l_t, c_t, f_t = None, None, None, None
        for network in self.networks:
            for layer in network.layers:
                if type(layer) is Conv2D:
                    weights = layer.get_weights()[0]
                    for c in range(weights.shape[2]):
                        for f in range(weights.shape[3]):
                            if i == filter1:
                                tmp, _ = layer.get_weights()
                                tmp = tmp[:, :, c, f]
                                n_t = network
                                l_t = layer
                                c_t = c
                                f_t = f
                            if i == filter2:
                                tmp2, b2 = layer.get_weights()
                                tmp2 = tmp2[:, :, c, f]
                                new_f = layer.get_weights()[0]
                                new_f[:, :, c, f] = tmp
                                layer.set_weights([new_f, b2])
                                for n in self.networks:
                                    if n == n_t:
                                        for l in n.layers:
                                            if l == l_t:
                                                wghts, b1 = l.get_weights()
                                                wghts[:, :, c_t, f_t] = tmp
                                                l.set_weights([wghts, b1])
                            i += 1
        return

    def apply_mutation(self):
        for i in range(len(self.networks)):
            if i == 0 and self.elitsm:
                pass
            else:
                for layer in self.networks[i].layers:
                    if type(layer) == Conv2D or type(layer) == Dense:
                        weights = layer.get_weights()
                        w = weights[0]
                        b = weights[1]
                        w += np.random.normal(scale=self.mutation,
                                              size=w.shape)
                        b += np.random.normal(scale=self.mutation,
                                              size=b.shape)
                        layer.set_weights([w, b])

if __name__ == '__main__':
    p = player.GeneticPlayer(psize=10)
    generations = 20
    val_iterations = 100

    idx = []
    win_percentages = []
    for z in tqdm(range(generations)):
        p.evolve(100)
        wins_p1 = 0.0
        wins_p2 = 0.0
        avg_moves_val = 0.0
        draws = 0.0
        p1 = p
        p2 = player.Player(2, False)
        test_game = Game(p1, p2)
        for j in range(val_iterations):
            (winner, moves) = test_game.play_game(False)
            avg_moves_val += moves
            if (winner == p1.value):
                wins_p1 += 1.0
            if (winner == p2.value):
                wins_p2 += 1.0
            elif (winner == 0):
                draws += 1.0
            test_game.reset_board()
            test_game.switch_players()
        win_percentages.append(wins_p1/val_iterations)
        idx.append(z)
    plt.plot(idx, win_percentages)
    plt.show()
    a = input('eind')
