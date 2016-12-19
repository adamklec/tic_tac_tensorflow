import tensorflow as tf
import numpy as np
import time
from collections import Counter
from random_agent import RandomAgent


class NeuralNetAgent(object):
    def __init__(self, env, sess, model_path, summary_path, checkpoint_path, restore=False):
        self.sess = sess
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.summary_path = summary_path
        self.env = env
        self.boards_placeholder = tf.placeholder(tf.float32, shape=[None, 3, 3, 2], name='candidate_boards')
        self.turn_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name='turn')
        reshaped_candidate_boards = tf.reshape(self.boards_placeholder, [tf.shape(self.boards_placeholder)[0], 18])
        feature_vectors = tf.concat(1, [reshaped_candidate_boards, self.turn_placeholder])
        with tf.variable_scope('layer_1'):
            W_1 = tf.Variable(tf.truncated_normal([19, 100], stddev=0.1), name='W_1')
            b_1 = tf.Variable(tf.constant(0.0, shape=[100]), name='b_1')
            activation_1 = tf.nn.relu(tf.matmul(feature_vectors, W_1) + b_1, name='activation_1')
        with tf.variable_scope('layer_2'):
            W_2 = tf.Variable(tf.truncated_normal([100, 1], stddev=0.1), name='W_2')
            b_2 = tf.Variable(tf.constant(0.0, shape=[1]), name='b_2')
            self.J = tf.nn.tanh(tf.matmul(activation_1, W_2) + b_2, name='J')

        self.J_next = tf.placeholder('float', [1, 1], name='J_next')
        delta_op = tf.reduce_sum(self.J_next - self.J, name='delta')

        turn_count = tf.Variable(tf.constant(0.0), name='turn_number', trainable=False)
        global_step = tf.Variable(0, trainable=False)
        self.increment_turn_count_op = turn_count.assign_add(1.0)
        self.increment_global_step_op = global_step.assign_add(1)
        self.turn_count_reset_op = turn_count.assign(0.0)
        tf.summary.scalar('global_step', global_step)

        self.batch_size_placeholder = tf.placeholder(tf.float32, [], name='batch_size')
        loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
        loss_op = tf.reduce_mean(tf.square(delta_op), name='loss')
        self.loss_sum_op = loss_sum.assign_add(loss_op)
        self.loss_avg_op = loss_sum / tf.maximum(turn_count, 1.0)
        self.loss_sum_reset_op = loss_sum.assign(0.0)
        tf.summary.scalar('loss_avg', self.loss_avg_op)
        tf.summary.scalar('average_turn_count', turn_count/self.batch_size_placeholder)

        lamda = tf.maximum(0.7, tf.train.exponential_decay(0.9, global_step, 30000, 0.96, staircase=True), name='lambda')
        tf.summary.scalar('lamda', lamda)

        update_traces = []
        update_trace_sums = []
        reset_trace_sums = []
        reset_traces = []
        trace_sums = []

        tvars = tf.trainable_variables()

        # learning_rate = tf.train.exponential_decay(0.01, global_step, 40000, 0.96, staircase=True, name='learning_rate')
        # opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        opt = tf.train.AdamOptimizer()
        grads_and_vars = opt.compute_gradients(self.J, var_list=tvars)
        # tf.summary.scalar('learning_rate', learning_rate)

        with tf.variable_scope('update_traces'):
            for grad, var in grads_and_vars:
                if grad is None:
                    grad = tf.zeros_like(var)
                with tf.variable_scope('trace'):
                    trace = tf.Variable(tf.zeros(var.get_shape()), trainable=False, name='trace')
                    update_trace_op = trace.assign(delta_op * ((lamda * trace) + grad))
                    update_traces.append(update_trace_op)

                    reset_trace_op = trace.assign(tf.zeros_like(trace))
                    reset_traces.append(reset_trace_op)

                trace_sum = tf.Variable(tf.zeros(var.get_shape()), trainable=False, name='trace_sum')
                trace_sums.append(trace_sum)

                update_trace_sum_op = trace_sum.assign_add(trace)
                update_trace_sums.append(update_trace_sum_op)
                reset_trace_sum_op = trace_sum.assign(tf.zeros_like(trace_sum))
                reset_trace_sums.append(reset_trace_sum_op)

        with tf.variable_scope('apply_traces'):
            self.apply_gradients_op = opt.apply_gradients(zip([-ts / turn_count for ts in trace_sums], tvars), global_step=global_step)

        for tvar, ts in zip(tvars, trace_sums):
            tf.summary.histogram(tvar.name, tvar)
            tf.summary.histogram(tvar.name + '/trace_sums', ts)

        self.update_traces_op = tf.group(*update_traces, name='update_traces')
        self.update_trace_sums_op = tf.group(*update_trace_sums, name='update_trace_sums')
        self.reset_trace_sums_op = tf.group(*reset_trace_sums, name='reset_trace_sums')
        self.reset_traces_op = tf.group(*reset_traces, name='reset_traces')

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

    def train(self, num_epochs, batch_size, epsilon, verbose=False):
        tf.train.write_graph(self.sess.graph_def, self.model_path, 'td_tictactoe.pb', as_text=False)
        summary_writer = tf.summary.FileWriter('{0}{1}'.format(self.summary_path, int(time.time())), graph=self.sess.graph)

        for epoch in range(num_epochs):
            for episode in range(batch_size):
                self.env.reset()
                while self.env.reward() is None:
                    candidate_boards = self.env.get_candidate_boards()
                    turns = float(not self.env.turn) * np.ones((len(candidate_boards), 1))
                    candidate_Js = self.sess.run(self.J, feed_dict={self.boards_placeholder: np.array(candidate_boards),
                                                                    self.turn_placeholder: turns})
                    if self.env.turn:
                        next_idx = np.argmax(candidate_Js)
                        next_J = np.max(candidate_Js)

                    else:
                        next_idx = np.argmin(candidate_Js)
                        next_J = np.min(candidate_Js)

                    self.sess.run([self.update_traces_op, self.loss_sum_op, self.increment_turn_count_op, self.increment_global_step_op],
                                  feed_dict={self.boards_placeholder: np.array([self.env.board]),
                                             self.turn_placeholder: np.array([[self.env.turn]]),
                                             self.J_next: np.array([[next_J]])})
                    if np.random.rand() < epsilon:
                        next_idx = np.random.choice(range(len(candidate_Js)))
                        self.sess.run([self.update_trace_sums_op])
                        self.sess.run([self.reset_traces_op])
                    self.env.step(candidate_boards[next_idx])

                self.sess.run([self.update_traces_op, self.loss_sum_op, self.increment_turn_count_op, self.increment_global_step_op],
                              feed_dict={self.boards_placeholder: np.array([self.env.board]),
                                         self.turn_placeholder: np.array([[self.env.turn]]),
                                         self.J_next: np.array([[self.env.reward()]])})
                self.sess.run([self.update_trace_sums_op])
                self.sess.run([self.reset_traces_op])

            if verbose:
                print('epoch', epoch)
                print('loss avg:', self.sess.run(self.loss_avg_op))
            self.test()
            self.sess.run(self.apply_gradients_op)
            self.saver.save(self.sess, self.checkpoint_path + 'checkpoint.ckpt')

            summary = self.sess.run(self.summaries_op, feed_dict={self.batch_size_placeholder: batch_size})
            summary_writer.add_summary(summary, epoch)
            self.sess.run([self.loss_sum_reset_op,
                           self.reset_trace_sums_op,
                           self.turn_count_reset_op])
        summary_writer.close()

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)

        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)

    def select_board(self, boards, turn):
        turns = float(self.env.turn) * np.ones((len(boards), 1))
        Js = self.sess.run(self.J, feed_dict={self.boards_placeholder: np.array(boards),
                                              self.turn_placeholder: turns})
        if turn:
            board_idx = Js.argmin()
        else:
            board_idx = Js.argmax()
        return boards[board_idx]

    def test(self):
        random_agent = RandomAgent()
        x_counter = Counter()
        for _ in range(100):
            self.env.reset()
            x_reward = self.env.play([self, random_agent])
            x_counter.update([x_reward])

        o_counter = Counter()
        for _ in range(100):
            self.env.reset()
            o_reward = self.env.play([random_agent, self])
            o_counter.update([o_reward])

        x_win_score = x_counter[1]*1.0/(x_counter[1]+x_counter[0]+x_counter[-1])
        o_win_score = o_counter[-1]*1.0/(o_counter[1]+o_counter[0]+o_counter[-1])

        self.sess.run([self.update_x_win_op, self.update_o_win_op], feed_dict={self.x_win_placeholder: x_win_score,
                                                                               self.o_win_placeholder: o_win_score})

        print('x rewards:', x_counter)
        print('o rewards:', o_counter)
        print(100 * '-')
