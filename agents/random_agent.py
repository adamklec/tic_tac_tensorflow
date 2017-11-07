import random
from collections import Counter


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_move(self):
        legal_moves = self.env.get_legal_moves()
        move = random.choice(legal_moves)
        return move

    def test(self, agent):
        x_counter = Counter()
        for _ in range(100):
            agent.env.reset()
            reward = agent.env.play([agent, self])
            x_counter.update([reward])

        o_counter = Counter()
        for _ in range(100):
            agent.env.reset()
            reward = agent.env.play([self, agent])
            o_counter.update([reward])

        results = [x_counter[1], x_counter[0], x_counter[-1],
                   o_counter[-1], o_counter[0], o_counter[1]]

        return results
