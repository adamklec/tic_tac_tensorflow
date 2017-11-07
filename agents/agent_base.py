import numpy as np
import tensorflow as tf


class AgentBase:

    def __init__(self, name, model, env):
        self.name = name
        self.model = model
        self.env = env
        self.sess = None

        self.episode_count = tf.train.get_or_create_global_step()
        self.increment_episode_count = tf.assign_add(self.episode_count, 1)

        for tvar in self.model.trainable_variables:
            tf.summary.histogram(tvar.op.name, tvar)

        with tf.name_scope('random_agent_test_results'):
            self.x_wins_ = tf.placeholder(tf.int32, name='x_wins_')
            self.x_wins = tf.Variable(0, name="x_wins", trainable=False)

            self.x_draws_ = tf.placeholder(tf.int32, name='x_draws_')
            self.x_draws = tf.Variable(0, name="x_draws", trainable=False)

            self.x_losses_ = tf.placeholder(tf.int32, name='x_losses_')
            self.x_losses = tf.Variable(0, name="x_losses", trainable=False)

            self.o_wins_ = tf.placeholder(tf.int32, name='o_wins_')
            self.o_wins = tf.Variable(0, name="o_wins", trainable=False)

            self.o_draws_ = tf.placeholder(tf.int32, name='o_draws_')
            self.o_draws = tf.Variable(0, name="o_draws", trainable=False)

            self.o_losses_ = tf.placeholder(tf.int32, name='o_losses_')
            self.o_losses = tf.Variable(0, name="o_losses", trainable=False)

            self.update_x_wins = tf.assign(self.x_wins, self.x_wins_)
            self.update_x_draws = tf.assign(self.x_draws, self.x_draws_)
            self.update_x_losses = tf.assign(self.x_losses, self.x_losses_)

            self.update_o_wins = tf.assign(self.o_wins, self.o_wins_)
            self.update_o_draws = tf.assign(self.o_draws, self.o_draws_)
            self.update_o_losses = tf.assign(self.o_losses, self.o_losses_)

            self.update_random_agent_test_results = tf.group(*[self.update_x_wins,
                                                               self.update_x_draws,
                                                               self.update_x_losses,
                                                               self.update_o_wins,
                                                               self.update_o_draws,
                                                               self.update_o_losses])
            self.random_agent_test_s = [self.x_wins_,
                                        self.x_draws_,
                                        self.x_losses_,
                                        self.o_wins_,
                                        self.o_draws_,
                                        self.o_losses_]

            tf.summary.scalar("x_wins", self.x_wins)
            tf.summary.scalar("x_draws", self.x_draws)
            tf.summary.scalar("x_losses", self.x_losses)

            tf.summary.scalar("o_wins", self.o_wins)
            tf.summary.scalar("o_draws", self.o_draws)
            tf.summary.scalar("o_losses", self.o_losses)

    def get_move(self):
        legal_moves = self.env.get_legal_moves()
        candidate_boards = []
        for move in legal_moves:
            candidate_board = self.env.board.copy()
            candidate_board.push(move)
            candidate_boards.append(candidate_board)

        feature_vectors = np.vstack(
            [self.env.make_feature_vector(board) for board in
             candidate_boards])

        values = self.sess.run(self.model.value,
                               feed_dict={
                                   self.model.feature_vector_: feature_vectors})

        for idx, board in enumerate(candidate_boards):
            result = board.result()
            if result is not None:
                values[idx] = result

        if self.env.board.turn:
            move_idx = np.argmax(values)
        else:
            move_idx = np.argmin(values)
        move = legal_moves[move_idx]

        return move
