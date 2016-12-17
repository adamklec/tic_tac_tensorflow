import numpy as np
from copy import copy


class TicTacToe(object):
    def __init__(self):
        self.board = np.zeros((3, 3, 2))
        self.turn = False

    def reset(self):
        self.board = np.zeros((3, 3, 2))
        self.turn = False
        return self.board, self.reward()

    def reward(self, board=None):
        if board is None:
            board = self.board
        if any(board[:, :, 0].sum(axis=0) == 3) or any(board[:, :, 0].sum(axis=1) == 3) or board[:, :, 0][np.eye(3) == 1].sum() == 3 or board[:, :, 0][np.rot90(np.eye(3)) == 1].sum() == 3:
            return 1
        elif any(board[:, :, 1].sum(axis=0) == 3) or any(board[:, :, 1].sum(axis=1) == 3) or board[:, :, 1][np.eye(3) == 1].sum() == 3 or board[:, :, 1][np.rot90(np.eye(3)) == 1].sum() == 3:
            return -1
        elif board.sum() == 9:
            return 0
        else:
            return None

    def step(self, board):
        self.board = board
        self.turn = not self.turn

    def get_candidate_boards(self, board=None):
        if board is None:
            board = self.board

        candidate_boards = []
        if self.reward() is None:
            empty_xs, empty_ys = np.where(board.sum(axis=2) == 0)
            for candidate_action in zip(empty_xs, empty_ys):
                candidate_board = copy(board)
                candidate_board[candidate_action[0], candidate_action[1], int(self.turn)] = 1
                candidate_boards.append(candidate_board)
        return candidate_boards

    def _print(self):
        s = ''
        for i in range(3):
            for j in range(3):
                if self.board[i, j, 0] == 1:
                    s += 'X'
                elif self.board[i, j, 1] == 1:
                    s += 'O'
                else:
                    s += '-'
            s += '\n'
        print(s)

    def play(self, players, verbose=False):
        while self.reward() is None:
            candidate_boards = self.get_candidate_boards()
            player = players[int(self.turn)]
            selected_board = player.select_board(candidate_boards, self.turn)
            self.step(selected_board)
            if verbose:
                self._print()
        return self.reward()
