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

    @abstractmethod
    def get_move(self, env):
        return NotImplemented

    def get_move_function(self):
        def m(env):
            move = self.get_move(env)
            return move
        return m
