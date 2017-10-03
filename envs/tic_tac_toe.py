import numpy as np
from .game_env_base import GameEnvBase
from .board_base import BoardBase
from copy import deepcopy
from random import choice, random


class TicTacToeEnv(GameEnvBase):
    def __init__(self):
        super().__init__()
        self.board = TicTacToeBoard()

    def reset(self):
        self.board = TicTacToeBoard()

    def random_position(self):
        self.reset()
        if random() > .5:
            legal_moves = self.get_legal_moves()
            move = choice(legal_moves)
            self.make_move(move)
            if random() > .75:
                legal_moves = self.get_legal_moves()
                move = choice(legal_moves)
                self.make_move(move)

    # def random_position(self):
    #     self.reset()
    #     move = np.random.randint(0, 18)
    #     legal_moves = self.get_legal_moves()
    #     if move in legal_moves:  # use starting position for moves greater than 8
    #         self.make_move(move)

    def get_reward(self, board=None):
        if board is None:
            board = self.board
        return board.result()

    def make_move(self, move):
        assert move in self.get_legal_moves()
        self.board.push(move)

    def get_legal_moves(self, board=None):
        if board is None:
            board = self.board
        return board.legal_moves

    @classmethod
    def make_feature_vector(cls, board):
        fv_size = cls.get_feature_vector_size()
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
        while self.get_reward() is None:
            if verbose:
                self._print()
            player = players[int(self.board.turn)]
            move = player.get_move(self)
            self.make_move(move)

        reward = self.get_reward()
        if verbose:
            self._print()
            if reward == 1:
                print("X won!")
            elif reward == -1:
                print("O won!")
            else:
                print("draw")
        return self.get_reward()

    @staticmethod
    def get_feature_vector_size():
        return 28


class TicTacToeBoard(BoardBase):
    def __init__(self, fen=None):
        super().__init__()

        self.xs = np.zeros((3, 3))
        self.os = np.zeros((3, 3))
        self._legal_moves = np.arange(9)
        self._turn = True
        self.move_stack = []

        if fen is not None:
            assert len(fen) == 9
            for i, char in enumerate(fen):
                row = int(i / 3)
                col = i % 3
                if char == 'X':
                    self.xs[row, col] = 1.0
                elif char == 'O':
                    self.xs[row, col] = 1.0
                else:
                    assert char == '-'
            self._turn = bool((self.xs.sum() + self.os.sum()) % 2)

    @property
    def turn(self):
        return self._turn

    @property
    def legal_moves(self):
        return self._legal_moves

    def fen(self):
        fen = ''
        for pair in zip(self.xs.reshape(9), self.os.reshape(9)):
            assert not (pair[0] and pair[1])
            if pair[0]:
                fen += 'X'
            elif pair[1]:
                fen += 'O'
            else:
                fen += '-'
        return fen

    def push(self, move):
        row = int(move / 3)
        col = move % 3

        assert self.xs[row, col] == 0
        assert self.os[row, col] == 0
        if self.turn:
            self.xs[row, col] = 1
        else:
            self.os[row, col] = 1
        self.move_stack.append(3 * row + col)
        self._turn = not self._turn
        self._legal_moves = np.where((self.xs + self.os).reshape(9) == 0)[0]

    def pop(self):
        move = self.move_stack[-1]
        self.move_stack = self.move_stack[:-1]

        row = int(move / 3)
        col = move % 3

        self.xs[row, col] = 0
        self.os[row, col] = 0

        return move

    def is_game_over(self):
        return self.result() is not None

    def result(self):
        if any(self.xs.sum(axis=0) == 3.0) or any(self.xs.sum(axis=1) == 3.0) or self.xs[np.eye(3) == 1.0].sum() == 3.0 or self.xs[np.rot90(np.eye(3)) == 1].sum() == 3.0:
            return 1.0
        elif any(self.os.sum(axis=0) == 3.0) or any(self.os.sum(axis=1) == 3.0) or self.os[np.eye(3) == 1.0].sum() == 3.0 or self.os[np.rot90(np.eye(3)) == 1].sum() == 3.0:
            return -1.0
        elif (self.xs + self.os).sum() == 9:
            return 0.0
        else:
            return None

    def copy(self):
        return deepcopy(self)
