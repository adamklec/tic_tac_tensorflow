import tensorflow as tf
import numpy as np
from agents.agent_base import AgentBase


class ForwardAgent(AgentBase):
    def __init__(self,
                 name,
                 model,
                 env,
                 verbose=False):

        super().__init__(name, model, env)

        self.opt = tf.train.AdamOptimizer()

        self.verbose = verbose

        self.grads = tf.gradients(self.model.value, self.model.trainable_variables)

        self.grads_s = [tf.placeholder(tf.float32, shape=tvar.get_shape()) for tvar in self.model.trainable_variables]

        self.apply_grads = self.opt.apply_gradients(zip(self.grads_s, self.model.trainable_variables),
                                                    name='apply_grads',
                                                    global_step=self.global_step_count)

    def train(self, epsilon):

        lamda = 0.7

        self.env.reset()

        grads_seq = []
        value_seq = []
        reward = self.env.get_reward()
        while reward is None:
            move, value = self.get_move(return_value=True)
            value_seq.append(value)

            if np.random.random() < epsilon:
                move = np.random.choice(self.env.get_legal_moves())
            self.env.make_move(move)

            reward = self.env.get_reward()

            feature_vector = self.env.make_feature_vector(self.env.board)
            grads = self.sess.run(self.grads, feed_dict={self.model.feature_vector_: feature_vector})
            grads_seq.append(grads)

        value_seq.append(reward)

        delta_seq = [j - i for i, j in zip(value_seq[:-1], value_seq[1:])]
        updates = [np.zeros(tvar.get_shape()) for tvar in self.model.trainable_variables]

        for t, grads in enumerate(grads_seq):
            for grad, update in zip(grads, updates):
                inner_sum = 0.0
                for j, delta in enumerate(delta_seq[t:]):
                    inner_sum += (lamda ** j) * delta
                update -= grad * inner_sum

        self.sess.run(self.apply_grads,
                      feed_dict={grad_: update/self.env.turn_count
                                 for grad_, update in zip(self.grads_s, updates)})

        return reward
