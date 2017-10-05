from abc import ABCMeta, abstractmethod
import tensorflow as tf


class AgentBase(metaclass=ABCMeta):

    def __init__(self, name, model, env, verbose=False):
        self.name = name
        self.model = model
        assign_tvar_ops = []
        for tvar, local_tvar in zip(self.model.trainable_variables, self.model.trainable_variables):
            assign_tvar_op = tf.assign(local_tvar, tvar)
            assign_tvar_ops.append(assign_tvar_op)
            tf.summary.histogram(tvar.op.name, tvar)
        self.env = env
        self.verbose = verbose
        self.sess = None

        self.global_step_count = tf.train.get_or_create_global_step()
        self.increment_global_step_count = tf.assign_add(self.global_step_count, 1)

    @abstractmethod
    def get_move(self, env):
        return NotImplemented

    def get_move_function(self):
        def m(env):
            move = self.get_move(env)
            return move
        return m
