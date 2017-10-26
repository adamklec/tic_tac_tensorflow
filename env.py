import numpy as np
from board import TicTacToeBoard
from random import choice


class TicTacToeEnv:
    def __init__(self):
        self.board = TicTacToeBoard()
        self.feature_vector_size = 28

    def reset(self):
        self.board = TicTacToeBoard()

    def get_reward(self):
        return self.board.result()

    def make_move(self, move):
        self.board.push(move)

    def make_random_move(self):
        legal_moves = self.get_legal_moves()
        move = choice(legal_moves)
        self.make_move(move)

    def get_legal_moves(self):
        return list(self.board.legal_moves)

    def make_feature_vector(self, board):
        fv_size = self.feature_vector_size
        fv = np.zeros((1, fv_size))
        fv[0, :9] = board.xs.reshape(9)
        fv[0, 9:18] = board.os.reshape(9)
        fv[0, 18:27] = ((board.xs + board.os).reshape(9) == 0)
        fv[0, -1] = float(board.turn)
        return fv

    def _print(self, board=None):
        if board is None:
            board = self.board
        s = ''
        for i in range(3):
            s += ' '
            for j in range(3):
                if board.xs[i, j] == 1:
                    s += 'X'
                elif board.os[i, j] == 1:
                    s += 'O'
                else:
                    s += ' '
                if j < 2:
                    s += '|'
            s += '\n'
            if i < 2:
                s += '-------\n'
        print(s)

    def play(self, players, verbose=False):
        reward = self.get_reward()
        while reward is None:
            if verbose:
                self._print()
            player = players[int(not self.board.turn)]
            move = player.get_move()
            self.make_move(move)
            reward = self.get_reward()

        if verbose:
            self._print()
            if reward == 1:
                print("X won!")
            elif reward == -1:
                print("O won!")
            else:
                print("draw.")
        return reward
