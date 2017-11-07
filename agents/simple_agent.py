import tensorflow as tf
import numpy as np
from agents.agent_base import AgentBase


class SimpleAgent(AgentBase):
    def __init__(self,
                 name,
                 model,
                 env,
                 verbose=False):

        AgentBase.__init__(self, name, model, env)

        self.opt = tf.train.AdamOptimizer()

        self.verbose = verbose

        self.value_ = tf.placeholder(tf.float32, name='value_')
        self.loss = tf.reduce_mean(tf.abs(self.value_ - self.model.value))
        self.train_op = self.opt.minimize(self.loss)

    def train(self, epsilon):

        self.env.reset()

        reward = None

        feature_vectors = []

        while reward is None:
            feature_vector = self.env.make_feature_vector(self.env.board)
            feature_vectors.append(feature_vector)

            if np.random.random() < epsilon:
                self.env.make_random_move()
            else:
                move = self.get_move()
                self.env.make_move(move)

            reward = self.env.get_reward()

        self.sess.run(self.train_op,
                      feed_dict={self.value_: reward,
                                 self.model.feature_vector_: np.vstack(feature_vectors)})

        return self.env.get_reward()
