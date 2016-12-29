import random


class RandomAgent(object):
    def get_move(self, env):
        legal_moves = env.get_legal_moves()
        move = random.choice(legal_moves)
        return move
