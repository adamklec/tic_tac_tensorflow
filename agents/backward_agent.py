import tensorflow as tf
import numpy as np
from agents.agent_base import AgentBase


class BackwardAgent(AgentBase):
    def __init__(self,
                 name,
                 model,
                 env,
                 verbose=False):

        AgentBase.__init__(self, name, model, env)

        self.verbose = verbose

        self.opt = tf.train.AdamOptimizer()

        self.grads = tf.gradients(self.model.value, self.model.trainable_variables)

        self.grads_s = [tf.placeholder(tf.float32, shape=tvar.get_shape()) for tvar in self.model.trainable_variables]

        self.apply_grads = self.opt.apply_gradients(zip(self.grads_s, self.model.trainable_variables),
                                                    name='apply_grads')

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
            for previous_grad, trace in zip(previous_grads, traces):
                trace *= lamda
                trace += previous_grad

            self.sess.run(self.apply_grads,
                          feed_dict={grad_: -delta * trace
                                     for grad_, trace in zip(self.grads_s, traces)})

            previous_grads = grads
            previous_value = value

        return self.env.get_reward()
