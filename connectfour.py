import numpy as np
import time
from tqdm import tqdm


class Game:

    def __init__(self, game=None):
        self.game = game  # The board
        self.pturn = 1  # Which player its turn it is (either 1 or -1)
        if game is None:
            self.game = np.zeros((6, 7), dtype=np.int8)

    def get_board(self):
        return Game(self.game.copy()).game * self.pturn

    def play(self, column):
        played = sum([abs(e) for e in self.game[:, column]])
        if played > 5:
            raise ValueError('The column is already full')
        self.game[5-played, column] = self.pturn
        self.pturn *= -1

    def end_state(self):
        # Check for horizontal
        for row in range(6):
            for col in range(4):
                if abs(self.game[row, col:col+4].sum()) == 4:
                    return True

        # Check for vertical
        for col in range(7):
            for row in range(3):
                if abs(self.game[row:row+4, col].sum()) == 4:
                    return True

        # Check for diagonal
        for col in range(4):
            for row in range(3):
                if abs(sum([self.game[row+i, col+i] for i in range(4)])) == 4:
                    return True
            for row in range(3, 6):
                if abs(sum([self.game[row-i, col+i] for i in range(4)])) == 4:
                    return True
        # Check if board is full otherwise return false
        return self.game.all()

if __name__ == '__main__':
    g = Game()
    for i in tqdm(range(100)):
        g = Game()
        while not g.end_state():
            a = np.random.randint(7)
            try:
                g.play(a)
            except ValueError:
                pass
    print g.game
