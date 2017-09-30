from abc import ABCMeta, abstractmethod
from agents.random_agent import RandomAgent
from collections import Counter
import tensorflow as tf


class GameEnvBase(metaclass=ABCMeta):
    def __init__(self):
        self.sess = None
        with tf.name_scope('random_agent_test_results'):
            self.first_player_wins_ = tf.placeholder(tf.int32, name='first_player_wins_')
            self.first_player_wins = tf.Variable(0, name="first_player_wins", trainable=False)

            self.first_player_draws_ = tf.placeholder(tf.int32, name='first_player_draws_')
            self.first_player_draws = tf.Variable(0, name="first_player_draws", trainable=False)

            self.first_player_losses_ = tf.placeholder(tf.int32, name='first_player_losses_')
            self.first_player_losses = tf.Variable(0, name="first_player_losses", trainable=False)

            self.second_player_wins_ = tf.placeholder(tf.int32, name='second_player_wins_')
            self.second_player_wins = tf.Variable(0, name="second_player_wins", trainable=False)

            self.second_player_draws_ = tf.placeholder(tf.int32, name='second_player_draws_')
            self.second_player_draws = tf.Variable(0, name="second_player_draws", trainable=False)

            self.second_player_losses_ = tf.placeholder(tf.int32, name='second_player_losses_')
            self.second_player_losses = tf.Variable(0, name="second_player_losses", trainable=False)

            self.update_first_player_wins = tf.assign(self.first_player_wins, self.first_player_wins_)
            self.update_first_player_draws = tf.assign(self.first_player_draws, self.first_player_draws_)
            self.update_first_player_losses = tf.assign(self.first_player_losses, self.first_player_losses_)

            self.update_second_player_wins = tf.assign(self.second_player_wins, self.second_player_wins_)
            self.update_second_player_draws = tf.assign(self.second_player_draws, self.second_player_draws_)
            self.update_second_player_losses = tf.assign(self.second_player_losses, self.second_player_losses_)

            self.update_random_agent_test_results = tf.group(*[self.update_first_player_wins,
                                                               self.update_first_player_draws,
                                                               self.update_first_player_losses,
                                                               self.update_second_player_wins,
                                                               self.update_second_player_draws,
                                                               self.update_second_player_losses])
            self.random_agent_test_s = [self.first_player_wins_,
                                        self.first_player_draws_,
                                        self.first_player_losses_,
                                        self.second_player_wins_,
                                        self.second_player_draws_,
                                        self.second_player_losses_]

            tf.summary.scalar("first_player_wins", self.first_player_wins)
            tf.summary.scalar("first_player_draws", self.first_player_draws)
            tf.summary.scalar("first_player_losses", self.first_player_losses)

            tf.summary.scalar("second_player_wins", self.second_player_wins)
            tf.summary.scalar("second_player_draws", self.second_player_draws)
            tf.summary.scalar("second_player_losses", self.second_player_losses)

    def getboard(self):
        return self.__board

    def setboard(self, value):
        self.__board = value

    board = property(getboard, setboard)

    @staticmethod
    @abstractmethod
    def get_feature_vector_size():
        return NotImplemented

    @abstractmethod
    def reset(self):
        return NotImplemented

    @abstractmethod
    def random_position(self):
        return NotImplemented

    @abstractmethod
    def get_reward(self, board=None):
        return NotImplemented

    @abstractmethod
    def make_move(self, move):
        return NotImplemented

    @abstractmethod
    def get_legal_moves(self, board=None):
        return NotImplemented

    @staticmethod
    @classmethod
    def make_feature_vector(cls, board):
        return NotImplemented

    def play_random(self, get_move_function, side):

        self.reset()
        random_agent = RandomAgent()
        if side:
            move_functions = [random_agent.get_move, get_move_function]  # True == 1 == 'X'
        else:
            move_functions = [get_move_function, random_agent.get_move]

        while self.get_reward() is None:
            move_function = move_functions[int(self.board.turn)]
            move = move_function(self)
            self.make_move(move)

        reward = self.get_reward()

        return reward

    def play_self(self, get_move_function):
        self.reset()
        while self.get_reward() is None:
            move = get_move_function(self)
            self.make_move(move)

        reward = self.get_reward()

        return reward

    def random_agent_test(self, get_move_function):
        x_counter = Counter()
        for _ in range(100):
            self.reset()
            reward = self.play_random(get_move_function, True)
            x_counter.update([reward])

        o_counter = Counter()
        for _ in range(100):
            self.reset()
            reward = self.play_random(get_move_function, False)
            o_counter.update([reward])

        results = [x_counter[1], x_counter[0], x_counter[-1],
                   o_counter[-1], o_counter[0], o_counter[1]]

        self.sess.run(self.update_random_agent_test_results, feed_dict={random_agent_test_: result
                                                                        for random_agent_test_, result in zip(self.random_agent_test_s, results)})
        return results
