import tensorflow as tf
import numpy as np
from agents.agent_base import AgentBase
from anytree import Node


class LeafAgent(AgentBase):
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

        previous_leaf_value, previous_grads = self.sess.run([self.model.value, self.grads],
                                                       feed_dict={self.model.feature_vector_: feature_vector})
        reward = self.env.get_reward()

        while reward is None:

            move, leaf_value, leaf_node = self.get_move(return_value=True)

            if np.random.rand() < epsilon:
                self.env.make_random_move()
            else:
                self.env.make_move(move)

            reward = self.env.get_reward()

            feature_vector = self.env.make_feature_vector(leaf_node.board)

            grads = self.sess.run(self.grads,
                                  feed_dict={self.model.feature_vector_: feature_vector})

            delta = leaf_value - previous_leaf_value
            for previous_grad, trace in zip(previous_grads, traces):
                trace *= lamda
                trace += previous_grad

            self.sess.run(self.apply_grads,
                          feed_dict={grad_: -delta * trace
                                     for grad_, trace in zip(self.grads_s, traces)})

            previous_grads = grads
            previous_leaf_value = leaf_value

        return self.env.get_reward()

    def minimax(self, node, depth, alpha, beta):

        if node.board.result() is not None:
            value = node.board.result()
            return np.array([[value]]), node

        elif depth <= 0:
            fv = self.env.make_feature_vector(node.board)
            value = self.sess.run(self.model.value, feed_dict={self.model.feature_vector_: fv})
            return value, node

        children = []
        for move in node.board.legal_moves:
            child_board = node.board.copy()
            child_board.push(move)
            child = Node(str(move), parent=node, board=child_board, move=move)
            children.append(child)

        if node.board.turn:
            best_v = -1
            best_n = None
            for child in children:
                value, node = self.minimax(child, depth - 1, alpha, beta)
                if value >= best_v:
                    best_v = value
                    best_n = node
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
        else:
            best_v = 1
            best_n = None
            for child in children:
                value, node = self.minimax(child, depth - 1, alpha, beta)
                if value <= best_v:
                    best_v = value
                    best_n = node
                beta = min(beta, value)
                if beta <= alpha:
                    break

        return best_v, best_n

    def get_move(self, depth=3, return_value=False):
        node = Node('root', board=self.env.board, move=None)
        leaf_value, leaf_node = self.minimax(node, depth, -1, 1)
        if len(leaf_node.path) > 1:
            move = leaf_node.path[1].move
        else:
            return self.env.get_null_move()

        if return_value:
            return move, leaf_value, leaf_node
        else:
            return move
