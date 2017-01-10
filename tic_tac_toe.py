import numpy as np
from copy import deepcopy


class TicTacToe(object):
    def __init__(self, board=None, turn=None):
        if board is None:
            self.board = np.zeros((3, 3, 3))
            self.board[:, :, 2] = 1
        else:
            self.board = board

        if turn is None:
            self.turn = False
        else:
            self.turn = turn

    def reset(self):
        self.board = np.zeros((3, 3, 3))
        self.board[:, :, 2] = 1
        self.turn = False

    def get_reward(self, board=None):
        if board is None:
            board = self.board
        if any(board[:, :, 0].sum(axis=0) == 3) or any(board[:, :, 0].sum(axis=1) == 3) or board[:, :, 0][np.eye(3) == 1].sum() == 3 or board[:, :, 0][np.rot90(np.eye(3)) == 1].sum() == 3:
            return 1
        elif any(board[:, :, 1].sum(axis=0) == 3) or any(board[:, :, 1].sum(axis=1) == 3) or board[:, :, 1][np.eye(3) == 1].sum() == 3 or board[:, :, 1][np.rot90(np.eye(3)) == 1].sum() == 3:
            return -1
        elif board[:, :, 2].sum() == 0:
            return 0
        else:
            return None

    def make_move(self, move):
        assert move in self.get_legal_moves()
        self.board[int(move / 3), move % 3, int(self.turn)] = 1
        self.board[int(move / 3), move % 3, 2] = 0
        self.turn = not self.turn

    def get_legal_moves(self, board=None):
        if board is None:
            board = self.board

        if self.get_reward() is None:
            legal_moves = np.where(board[:, :, 2].flatten() == 1)[0]
        else:
            legal_moves = np.array([])
        return legal_moves

    def get_candidate_boards(self):
        legal_moves = self.get_legal_moves()
        candidate_boards = []
        for legal_move in legal_moves:
            candidate_board = deepcopy(self.board)
            candidate_board[int(legal_move / 3), legal_move % 3, int(self.turn)] = 1
            candidate_board[int(legal_move / 3), legal_move % 3, 2] = 0
            candidate_boards.append(candidate_board)
        return candidate_boards

    def _print(self, board=None):
        if board is None:
            board = self.board
        s = ''
        for i in range(3):
            s += ' '
            for j in range(3):
                if board[i, j, 0] == 1:
                    s += 'X'
                elif board[i, j, 1] == 1:
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
            player = players[int(self.turn)]
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

    def clone(self):
        return TicTacToe(board=deepcopy(self.board), turn=deepcopy(self.turn))
