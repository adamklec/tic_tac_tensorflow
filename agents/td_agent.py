import tensorflow as tf
import numpy as np
from agents.agent_base import AgentBase


class TDAgent(AgentBase):
    def __init__(self,
                 name,
                 model,
                 env,
                 verbose=False):

        AgentBase.__init__(self, name, model, env)

        self.opt = tf.train.AdamOptimizer()

        self.verbose = verbose

        self.grads = tf.gradients(self.model.value, self.model.trainable_variables)

        self.grads_s = [tf.placeholder(tf.float32, shape=tvar.get_shape()) for tvar in self.model.trainable_variables]

        self.apply_grads = self.opt.apply_gradients(zip(self.grads_s, self.model.trainable_variables),
                                                    name='apply_grads')

    def train(self, epsilon):

        self.env.reset()

        feature_vector = self.env.make_feature_vector(self.env.board)

        previous_value, previous_grads = self.sess.run([self.model.value, self.grads],
                                                       feed_dict={self.model.feature_vector_: feature_vector})
        reward = self.env.get_reward()

        while reward is None:

            if np.random.random() < epsilon:
                self.env.make_random_move()
            else:
                move = self.get_move()
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
            self.sess.run(self.apply_grads,
                          feed_dict={grad_: -delta * previous_grad
                                     for previous_grad, grad_ in zip(previous_grads, self.grads_s)})

            previous_grads = grads
            previous_value = value

        return self.env.get_reward()
