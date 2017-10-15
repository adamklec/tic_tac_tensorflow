import numpy as np
from copy import deepcopy


class TicTacToeBoard:
    def __init__(self):
        self.xs = np.zeros((3, 3))
        self.os = np.zeros((3, 3))
        self.legal_moves = set(range(9))
        self.turn = True

    def push(self, move):
        row = int(move / 3)
        col = move % 3
        assert self.xs[row, col] == 0
        assert self.os[row, col] == 0

        assert move in self.legal_moves
        self.legal_moves.remove(move)

        if self.turn:
            self.xs[row, col] = 1
        else:
            self.os[row, col] = 1
        self.turn = not self.turn

    def result(self):
        if self.xs.all(axis=0).any() or self.xs.all(axis=1).any() or self.xs.diagonal().all() or np.rot90(self.xs).diagonal().all():
            return 1.0
        elif self.os.all(axis=0).any() or self.os.all(axis=1).any() or self.os.diagonal().all() or np.rot90(self.os).diagonal().all():
            return -1.0
        elif (self.xs + self.os).sum() == 9:
            return 0.0
        else:
            return None

    def copy(self):
        return deepcopy(self)
