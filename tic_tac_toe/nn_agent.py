import numpy as np


class NeuralNetworkAgent(object):
    def __init__(self, model):
        self.model = model

    def get_move(self, env):
        legal_moves = env.get_legal_moves()
        candidate_boards = env.get_candidate_boards()
        candidate_Js = self.model.calculate_J(candidate_boards, env.turn)

        if env.turn:
            move_idx = np.argmin(candidate_Js)
        else:
            move_idx = np.argmax(candidate_Js)
        return legal_moves[move_idx]
