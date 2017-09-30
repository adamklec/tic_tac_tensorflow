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
                                                    global_step=self.global_episode_count)

        ema = tf.train.ExponentialMovingAverage(decay=0.9999)

        delta = tf.Variable(0.0, trainable=False, name='mean_delta')
        self.delta_ = tf.placeholder(tf.float32, name='delta_')
        assign_delta = tf.assign(delta, tf.abs(self.delta_))

        with tf.control_dependencies([assign_delta]):
            self.update_delta = ema.apply([delta])

        tf.summary.scalar("mean_delta", ema.average(delta))

        layer_1_grad_accum_norm = tf.Variable(0.0, trainable=False, name='layer_1_grad_accum_norm')
        self.layer_1_grad_accum_norm_ = tf.placeholder(tf.float32, name='layer_1_grad_accum_norm_')
        self.update_layer_1_grad_accum_norm = tf.assign(layer_1_grad_accum_norm, self.layer_1_grad_accum_norm_)
        tf.summary.scalar("layer_1_grad_accum_norm", layer_1_grad_accum_norm)


        layer_2_grad_accum_norm = tf.Variable(0.0, trainable=False, name='layer_2_grad_accum_norm')
        self.layer_2_grad_accum_norm_ = tf.placeholder(tf.float32, name='layer_2_grad_accum_norm_')
        self.update_layer_2_grad_accum_norm = tf.assign(layer_2_grad_accum_norm, self.layer_2_grad_accum_norm_)
        tf.summary.scalar("layer_2_grad_accum_norm", layer_2_grad_accum_norm)

    def train(self, epsilon):

        lamda = 0.7

        self.env.random_position()

        traces = [np.zeros(tvar.shape)
                  for tvar in self.model.trainable_variables]
        grad_accums = [np.zeros(tvar.shape)
                       for tvar in self.model.trainable_variables]
        turn_count = 0
        previous_grads = None
        previous_value = None
        while self.env.get_reward() is None:

            move, value = self.get_move(self.env, return_value=True)
            if np.random.random() < epsilon:
                move = np.random.choice(self.env.get_legal_moves())
            self.env.make_move(move)

            feature_vector = self.env.make_feature_vector(self.env.board)
            grads = self.sess.run(self.grads, feed_dict={self.model.feature_vector_: feature_vector})

            if turn_count > 0:
                delta = (value - previous_value)
                for previous_grad, trace, grad_accum in zip(previous_grads, traces, grad_accums):
                    trace *= lamda
                    trace += previous_grad
                    grad_accum -= delta * trace
                self.sess.run(self.update_delta,
                              feed_dict={self.delta_: delta})

            previous_grads = grads
            previous_value = value
            turn_count += 1

        self.sess.run(self.apply_grads,
                      feed_dict={grad_: grad_accum
                                 for grad_, grad_accum in zip(self.grads_s, grad_accums)})

        self.sess.run([self.update_layer_1_grad_accum_norm,
                       self.update_layer_2_grad_accum_norm],
                      feed_dict={self.layer_1_grad_accum_norm_: np.linalg.norm(grad_accums[0]),
                                 self.layer_2_grad_accum_norm_: np.linalg.norm(grad_accums[1])
                                 }
                      )

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
            value = np.max(values)
            move_idx = np.argmax(values)
        else:
            value = np.min(values)
            move_idx = np.argmin(values)
        move = env.get_legal_moves()[move_idx]
        if return_value:
            return move, value
        else:
            return move

    def get_move_function(self):
        def m(env):
            move = self.get_move(env)
            return move
        return m
