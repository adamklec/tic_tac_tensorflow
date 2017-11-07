import tensorflow as tf
import numpy as np
from agents.agent_base import AgentBase


class ForwardAgent(AgentBase):
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

        lamda = 0.7

        self.env.reset()

        grads_seq = []
        value_seq = []
        reward = self.env.get_reward()

        while reward is None:
            feature_vector = self.env.make_feature_vector(self.env.board)
            value, grads = self.sess.run([self.model.value, self.grads],
                                         feed_dict={self.model.feature_vector_: feature_vector})
            value_seq.append(value)
            grads_seq.append(grads)

            if np.random.random() < epsilon:
                self.env.make_random_move()
            else:
                move = self.get_move()
                self.env.make_move(move)

            reward = self.env.get_reward()

        value_seq.append(np.array([reward]))

        delta_seq = np.array([j - i for i, j in zip(value_seq[:-1], value_seq[1:])])
        # delta_seq[:-1] = delta_seq[:-1] * (1.0 - lamda)

        for t, grads in enumerate(grads_seq):
            delta_sum = np.sum([(lamda ** j) * delta for j, delta in enumerate(delta_seq[t:])])
            self.sess.run(self.apply_grads,
                          feed_dict={grad_: -grad * delta_sum
                                     for grad_, grad in zip(self.grads_s, grads)})

        return reward
