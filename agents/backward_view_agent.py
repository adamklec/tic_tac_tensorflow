import tensorflow as tf
import numpy as np
from agents.agent_base import AgentBase


class BackwardViewAgent(AgentBase):
    def __init__(self,
                 name,
                 model,
                 env,
                 verbose=False):

        super().__init__(name, model, env, verbose)

        self.opt = tf.train.AdamOptimizer()

        self.grads = tf.gradients(self.model.value, self.model.trainable_variables)

        self.grads_s = [tf.placeholder(tf.float32, shape=tvar.get_shape()) for tvar in self.model.trainable_variables]

        self.apply_grads = self.opt.apply_gradients(zip(self.grads_s, self.model.trainable_variables),
                                                    name='apply_grads',
                                                    global_step=self.global_step_count)

    def train(self, epsilon):

        lamda = 0.7

        self.env.reset()

        traces = [np.zeros(tvar.shape)
                  for tvar in self.model.trainable_variables]

        feature_vector = self.env.make_feature_vector(self.env.board)

        previous_value, previous_grads = self.sess.run([self.model.value, self.grads],
                                                       feed_dict={self.model.feature_vector_: feature_vector})
        reward = self.env.get_reward()

        while reward is None:

            move = self.get_move(self.env)
            if np.random.random() < epsilon:
                move = np.random.choice(self.env.get_legal_moves())
            self.env.make_move(move)

            reward = self.env.get_reward()

            feature_vector = self.env.make_feature_vector(self.env.board)

            if reward is None:
                value, grads = self.sess.run([self.model.value, self.grads],
                                             feed_dict={self.model.feature_vector_: feature_vector})
            else:
                value = reward
                grads = self.sess.run(self.grads,
                                      feed_dict={self.model.feature_vector_: feature_vector})

            delta = value - previous_value
            for previous_grad, trace in zip(previous_grads, traces):
                trace *= lamda
                trace += previous_grad

            self.sess.run(self.apply_grads,
                          feed_dict={grad_: -delta * trace
                                     for grad_, trace in zip(self.grads_s, traces)})

            previous_grads = grads
            previous_value = value

        return self.env.get_reward()

    def get_move(self, env, return_value=False):
        legal_moves = env.get_legal_moves()
        candidate_boards = []
        for move in legal_moves:
            candidate_board = self.env.board.copy()
            candidate_board.push(move)
            candidate_boards.append(candidate_board)

        feature_vectors = np.vstack(
            [self.env.make_feature_vector(board) for board in
             candidate_boards])

        values = self.sess.run(self.model.value,
                               feed_dict={self.model.feature_vector_: feature_vectors})

        for idx, board in enumerate(candidate_boards):
            result = board.result()
            if result is not None:
                values[idx] = result

        if env.board.turn:
            move_idx = np.argmax(values)
        else:
            move_idx = np.argmin(values)
        move = legal_moves[move_idx]

        return move
