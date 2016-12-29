import tensorflow as tf
import numpy as np
import time
from collections import Counter
from agents.random_agent import RandomAgent
from random import choice


class NeuralNetworkAgent(object):
    def __init__(self, sess, model_path, summary_path, checkpoint_path, restore=False):
        self.sess = sess
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.summary_path = summary_path

        game_turn_count = tf.Variable(0, name='game_turn_count', trainable=False, dtype=tf.int32)
        batch_turn_count = tf.Variable(0, name='batch_turn_count', trainable=False, dtype=tf.int32)
        global_turn_count = tf.Variable(0, name='global_turn_count', trainable=False, dtype=tf.int32)
        self.increment_turn_count_op = tf.group(game_turn_count.assign_add(1),
                                                batch_turn_count.assign_add(1),
                                                global_turn_count.assign_add(1))
        self.reset_game_turn_count_op = game_turn_count.assign(0)
        self.reset_batch_turn_count_op = batch_turn_count.assign(0)
        self.reset_global_turn_count_op = global_turn_count.assign(0)
        tf.summary.scalar('global_turn_count', global_turn_count)

        self.batch_size_placeholder = tf.placeholder(tf.float32, shape=[], name='batch_size_placeholder')
        self.mean_turn_count_per_game = tf.cast(batch_turn_count, tf.float32)/self.batch_size_placeholder
        tf.summary.scalar('meanturn_count_per_game', self.mean_turn_count_per_game)

        self.turn_placeholder = tf.placeholder(tf.bool, shape=[], name='turn')

        self.board_placeholder = tf.placeholder(tf.float32, shape=[1, 3, 3, 3], name='board_placeholder')
        feature_vector = self.make_feature_vectors(self.board_placeholder, self.turn_placeholder)
        value = self.neural_net(feature_vector)

        self.next_boards_placeholder = tf.placeholder(tf.float32, shape=[None, 3, 3, 3], name='next_boards')
        feature_vectors = self.make_feature_vectors(self.next_boards_placeholder, tf.logical_not(self.turn_placeholder))
        next_values = self.neural_net(feature_vectors)
        next_value = tf.cond(self.turn_placeholder, lambda: tf.reduce_min(next_values), lambda: tf.reduce_max(next_values))
        self.next_board_idx = tf.cond(self.turn_placeholder, lambda: tf.argmin(next_values, axis=0), lambda: tf.argmax(next_values, axis=0))
        # next_board = tf.slice(self.next_boards_placeholder, [next_board_idx, 0, 0, 0], [1, 3, 3, 3])

        self.reward_placeholder = tf.placeholder(tf.float32, shape=[], name='reward_placeholder')

        target_value = tf.cond(tf.shape(self.next_boards_placeholder)[0] > 0, lambda: next_value, lambda: self.reward_placeholder)
        delta = tf.reduce_sum(target_value - value)
        loss = tf.reduce_mean(tf.square(target_value - value, name='loss'))

        self.batch_loss = tf.Variable(0.0, trainable=False, name='loss_sum')
        self.mean_step_loss = self.batch_loss / tf.maximum(1.0, tf.cast(batch_turn_count, tf.float32))
        self.update_batch_loss_op = self.batch_loss.assign_add(loss)
        self.reset_batch_loss_op = self.batch_loss.assign(0.0)

        tf.summary.scalar('mean_step_loss', self.mean_step_loss)

        tvars = tf.trainable_variables()
        opt = tf.train.AdamOptimizer()
        grads_and_vars = opt.compute_gradients(value, var_list=tvars)

        lamda = tf.constant(0.7, name='lamba')
        tf.summary.scalar('lamda', lamda)

        grad_traces = []
        update_grad_traces = []
        reset_grad_traces = []

        grad_trace_sums = []
        update_grad_trace_sums = []
        reset_grad_trace_sums = []

        with tf.variable_scope('update_traces'):
            for grad, var in grads_and_vars:
                if grad is None:
                    grad = tf.zeros_like(var)
                with tf.variable_scope('trace'):
                    grad_trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='grad_trace')
                    grad_traces.append(grad_trace)

                    update_grad_trace_op = grad_trace.assign_add(-delta * ((lamda * grad_trace) + grad)) #negaive sign for optimizer
                    update_grad_traces.append(update_grad_trace_op)

                    reset_grad_trace_op = grad_trace.assign(tf.zeros_like(grad_trace))
                    reset_grad_traces.append(reset_grad_trace_op)

                    grad_trace_sum = tf.Variable(tf.zeros(var.get_shape()), trainable=False, name='grad_trace')
                    grad_trace_sums.append(grad_trace_sum)

                    update_grad_trace_sum_op = grad_trace_sum.assign_add(grad_trace)
                    update_grad_trace_sums.append(update_grad_trace_sum_op)

                    reset_grad_trace_sum_op = grad_trace.assign(tf.zeros_like(grad_trace_sum))
                    reset_grad_trace_sums.append(reset_grad_trace_sum_op)

        for tvar, grad_trace_sum in zip(tvars, grad_trace_sums):
            tf.summary.histogram(tvar.name, tvar)
            tf.summary.histogram(tvar.name + '/grad_traces_sum', grad_trace_sum)

        self.update_grad_traces_op = tf.group(*update_grad_traces)
        self.reset_grad_traces_op = tf.group(*reset_grad_traces)

        self.update_grad_trace_sums_op = tf.group(*update_grad_trace_sums)
        self.reset_grad_trace_sums_op = tf.group(*reset_grad_trace_sums)

        self.apply_grad_trace_sums_op = opt.apply_gradients(zip(grad_trace_sums, tvars), global_step=global_turn_count)

        xwin = tf.Variable(tf.constant(0.0), name='xwin', trainable=False)
        self.x_win_placeholder = tf.placeholder(tf.float32, shape=[], name='x_win_placeholder')
        self.update_x_win_op = tf.assign(xwin, self.x_win_placeholder)
        tf.summary.scalar('x_win', xwin)

        owin = tf.Variable(tf.constant(0.0), name='owin', trainable=False)
        self.o_win_placeholder = tf.placeholder(tf.float32, shape=[], name='o_win_placeholder')
        self.update_o_win_op = tf.assign(owin, self.o_win_placeholder)
        tf.summary.scalar('o_win', owin)

        self.summaries_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(var_list=tvars, max_to_keep=1)

        self.sess.run(tf.global_variables_initializer())

        if restore:
            self.restore()

    def make_feature_vectors(self, boards, turn):
        turn = tf.reshape(tf.cast(turn, tf.float32), [-1, 1])
        turns = tf.tile(turn, [1, tf.shape(boards)[0]])
        turns = tf.transpose(turns)

        reshaped_candidate_next_boards = tf.reshape(boards, [tf.shape(boards)[0], 27])
        feature_vectors = tf.concat(1, [reshaped_candidate_next_boards, turns])

        return feature_vectors

    def neural_net(self, feature_vector):
        with tf.variable_scope("value_function") as scope:
            with tf.variable_scope('layer_1'):
                W_1 = tf.Variable(tf.truncated_normal([28, 100], stddev=0.1), name='W_1')
                b_1 = tf.Variable(tf.constant(0.0, shape=[100]), name='b_1')
                activation_1 = tf.nn.relu(tf.matmul(feature_vector, W_1) + b_1, name='activation_1')
            with tf.variable_scope('layer_2'):
                W_2 = tf.Variable(tf.truncated_normal([100, 1], stddev=0.1), name='W_2')
                b_2 = tf.Variable(tf.constant(0.0, shape=[1]), name='b_2')
                value = tf.nn.tanh(tf.matmul(activation_1, W_2) + b_2, name='J')
                return value

    def train(self, env, num_epochs, batch_size, epsilon, run_name=None, verbose=False):
        tf.train.write_graph(self.sess.graph_def, self.model_path, 'td_tictactoe.pb', as_text=False)
        if run_name is None:
            summary_writer = tf.summary.FileWriter('{0}{1}'.format(self.summary_path, int(time.time())), graph=self.sess.graph)
        else:
            summary_writer = tf.summary.FileWriter('{0}{1}'.format(self.summary_path, run_name), graph=self.sess.graph)

        for epoch in range(num_epochs):
            if verbose:
                print('epoch', epoch)
            for episode in range(batch_size):
                env.reset()
                while env.reward() is None:
                    legal_moves = env.get_legal_moves()
                    move_idx, _, _, _ = self.sess.run([self.next_board_idx, self.update_grad_traces_op, self.increment_turn_count_op, self.update_batch_loss_op],
                                                      feed_dict={self.turn_placeholder: env.turn,
                                                                 self.board_placeholder: [env.board],
                                                                 self.next_boards_placeholder: env.get_candidate_boards(),
                                                                 self.reward_placeholder: 0.0})
                    move = legal_moves[move_idx]

                    if np.random.rand() < epsilon:
                        move = choice(legal_moves)
                        self.sess.run(self.update_grad_trace_sums_op)
                        self.sess.run(self.reset_grad_traces_op)

                    env.make_move(move)

                self.sess.run([self.update_grad_traces_op, self.update_batch_loss_op],
                              feed_dict={self.turn_placeholder: int(env.turn),
                                         self.board_placeholder: [env.board],
                                         self.next_boards_placeholder: np.zeros((0, 3, 3, 3)),
                                         self.reward_placeholder: env.reward()})
                self.sess.run([self.update_grad_trace_sums_op])
                self.sess.run([self.reset_grad_traces_op, self.reset_game_turn_count_op])

            self.sess.run(self.apply_grad_trace_sums_op)

            if verbose:
                print('mean_step_loss:', self.sess.run(self.mean_step_loss))

            self.test(env)

            self.saver.save(self.sess, self.checkpoint_path + 'checkpoint.ckpt')
            summary = self.sess.run(self.summaries_op, feed_dict={self.batch_size_placeholder: batch_size})
            summary_writer.add_summary(summary, epoch)

            self.sess.run([self.reset_batch_loss_op,
                           self.reset_grad_trace_sums_op,
                           self.reset_batch_turn_count_op])

        summary_writer.close()

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)

        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)

    def test(self, env):
        random_agent = RandomAgent()

        x_counter = Counter()
        for _ in range(100):
            env.reset()
            x_reward = env.play([self, random_agent])
            x_counter.update([x_reward])

        o_counter = Counter()
        for _ in range(100):
            env.reset()
            o_reward = env.play([random_agent, self])
            o_counter.update([o_reward])

        x_win_score = x_counter[1]*1.0/(x_counter[1]+x_counter[0]+x_counter[-1])
        o_win_score = o_counter[-1]*1.0/(o_counter[1]+o_counter[0]+o_counter[-1])

        self.sess.run([self.update_x_win_op, self.update_o_win_op],
                      feed_dict={self.x_win_placeholder: x_win_score,
                                 self.o_win_placeholder: o_win_score})

        print('x rewards:', x_counter)
        print('o rewards:', o_counter)
        print(100 * '-')

    def get_move(self, env):
        legal_moves = env.get_legal_moves()
        move_idx = self.sess.run(self.next_board_idx,
                                 feed_dict={self.turn_placeholder: env.turn,
                                            self.next_boards_placeholder: env.get_candidate_boards()})
        return legal_moves[move_idx]


