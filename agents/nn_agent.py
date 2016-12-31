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

        self.batch_size_placeholder = tf.placeholder(tf.float32, shape=[], name='batch_size_placeholder')

        with tf.variable_scope('turn_count'):
            game_turn_count = tf.Variable(0, name='game_turn_count', trainable=False, dtype=tf.int32)
            batch_turn_count = tf.Variable(0, name='batch_turn_count', trainable=False, dtype=tf.int32)
            global_turn_count = tf.Variable(0, name='global_turn_count', trainable=False, dtype=tf.int32)
            self.increment_turn_count_op = tf.group(game_turn_count.assign_add(1),
                                                    batch_turn_count.assign_add(1),
                                                    global_turn_count.assign_add(1))
            self.reset_game_turn_count_op = game_turn_count.assign(0)
            self.reset_batch_turn_count_op = batch_turn_count.assign(0)
            self.reset_global_turn_count_op = global_turn_count.assign(0)
            # self.mean_turn_count_per_game = tf.cast(batch_turn_count, tf.float32)/self.batch_size_placeholder

        # tf.summary.scalar('mean_turn_count_per_game', self.mean_turn_count_per_game)

        self.turn_placeholder = tf.placeholder(tf.bool, shape=[], name='turn')
        self.board_placeholder = tf.placeholder(tf.float32, shape=[1, 3, 3, 3], name='board_placeholder')
        self.next_boards_placeholder = tf.placeholder(tf.float32, shape=[None, 3, 3, 3], name='next_boards')

        with tf.variable_scope('feature_vectors'):
            feature_vector = self.make_feature_vectors(self.board_placeholder, self.turn_placeholder)
            feature_vectors = self.make_feature_vectors(self.next_boards_placeholder, tf.logical_not(self.turn_placeholder))

        with tf.variable_scope('state_value_function') as scope:
            value = self.neural_network(feature_vector)
            scope.reuse_variables()
            next_values = self.neural_network(feature_vectors)

        next_value = tf.cond(self.turn_placeholder, lambda: tf.reduce_min(next_values), lambda: tf.reduce_max(next_values), name='next_value')
        self.next_board_idx = tf.cond(self.turn_placeholder, lambda: tf.argmin(next_values, axis=0), lambda: tf.argmax(next_values, axis=0), name='next_board_idx')
        # next_board = tf.slice(self.next_boards_placeholder, [next_board_idx, 0, 0, 0], [1, 3, 3, 3])

        self.reward_placeholder = tf.placeholder(tf.float32, shape=[], name='reward_placeholder')

        target_value = tf.cond(tf.shape(self.next_boards_placeholder)[0] > 0, lambda: next_value, lambda: self.reward_placeholder, name='target_value')
        delta = tf.sub(target_value, value, name='delta')
        loss = tf.reduce_sum(tf.square(delta, name='loss'))

        tvars = tf.trainable_variables()
        opt = tf.train.AdamOptimizer()
        grads_and_vars = opt.compute_gradients(value, var_list=tvars)

        lamda = tf.constant(0.7, name='lamba')
        # tf.summary.scalar('lamda', lamda)

        traces = []
        update_traces = []
        reset_traces = []

        with tf.variable_scope('update_traces'):
            for grad, var in grads_and_vars:
                if grad is None:
                    grad = tf.zeros_like(var)
                with tf.variable_scope('trace'):
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    traces.append(trace)

                    update_trace_op = trace.assign((lamda * trace) + grad)
                    update_traces.append(update_trace_op)

                    reset_trace_op = trace.assign(tf.zeros_like(trace))
                    reset_traces.append(reset_trace_op)

        for tvar, trace in zip(tvars, traces):
            tf.summary.histogram(tvar.name, tvar)
            # tf.summary.histogram(tvar.name + '/delta_trace', tf.reduce_sum(delta) * trace)

        self.update_traces_op = tf.group(*update_traces)
        self.reset_traces_op = tf.group(*reset_traces)

        self.average_loss = tf.Variable(0.0, trainable=False)
        loss_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        average_loss_update_op = tf.group(loss_ema.apply([loss]),
                                          self.average_loss.assign(loss_ema.average(loss)))
        tf.summary.scalar('average_loss', self.average_loss)

        average_turn_count_per_game = tf.Variable(0.0, trainable=False)
        turn_count_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        game_turn_count_m1 = tf.cast(game_turn_count, tf.float32) - 1.0

        # TODO: control dependencies
        self.average_turn_count_per_game_update_op = tf.group(turn_count_ema.apply([game_turn_count_m1]),
                                                              average_turn_count_per_game.assign(turn_count_ema.average(game_turn_count_m1)))
        tf.summary.scalar('average_turn_count_per_game', average_turn_count_per_game)

        with tf.control_dependencies([self.increment_turn_count_op,
                                      self.update_traces_op,
                                      average_loss_update_op]):
            # negative sign for gradient ascent
            self.train_op = opt.apply_gradients(zip([-tf.reduce_sum(delta) * trace / tf.cast(batch_turn_count, tf.float32)
                                                     for trace in traces],
                                                    tvars))

        xwin = tf.Variable(tf.constant(0.0), name='xwin', trainable=False)
        self.x_win_placeholder = tf.placeholder(tf.float32, shape=[], name='x_win_placeholder')
        self.update_x_win_op = tf.assign(xwin, self.x_win_placeholder)
        tf.summary.scalar('x_win', xwin)

        owin = tf.Variable(tf.constant(0.0), name='owin', trainable=False)
        self.o_win_placeholder = tf.placeholder(tf.float32, shape=[], name='o_win_placeholder')
        self.update_o_win_op = tf.assign(owin, self.o_win_placeholder)
        tf.summary.scalar('o_win', owin)

        self.saver = tf.train.Saver(var_list=tvars, max_to_keep=1)

        self.sess.run(tf.global_variables_initializer())

        for tvar in tvars:
            for slot_name in opt.get_slot_names():
                slot = opt.get_slot(tvar, slot_name)
                tf.summary.histogram(tvar.name + '/' + slot_name, slot)

        # TODO: break up summaries
        self.summaries_op = tf.summary.merge_all()

        if restore:
            self.restore()

    def make_feature_vectors(self, boards, turn):
        with tf.variable_scope('make_feature_vectors'):
            turn = tf.reshape(tf.cast(turn, tf.float32), [-1, 1])
            turns = tf.tile(turn, [1, tf.shape(boards)[0]])
            turns = tf.transpose(turns)

            reshaped_candidate_next_boards = tf.reshape(boards, [tf.shape(boards)[0], 27])
            feature_vectors = tf.concat(1, [reshaped_candidate_next_boards, turns])

            return feature_vectors

    def neural_network(self, feature_vector):

        with tf.variable_scope("neural_network"):
            with tf.variable_scope('layer_1'):
                W_1 = tf.get_variable('W_1', initializer=tf.truncated_normal([28, 100], stddev=0.1))
                b_1 = tf.get_variable('b_1', shape=[100], initializer=tf.constant_initializer(-0.1))
                activation_1 = tf.nn.relu(tf.matmul(feature_vector, W_1) + b_1, name='activation_1')

            with tf.variable_scope('layer_2'):
                W_2 = tf.get_variable('W_2', initializer=tf.truncated_normal([100, 1], stddev=0.1))
                b_2 = tf.get_variable('b_2', shape=[1],  initializer=tf.constant_initializer(0.0))
                value = tf.nn.tanh(tf.matmul(activation_1, W_2) + b_2, name='value')

            return value

    def train(self, env, num_epochs, batch_size, epsilon, run_name=None, verbose=False, summary_interval = 100):
        tf.train.write_graph(self.sess.graph_def, self.model_path, 'td_tictactoe.pb', as_text=False)
        if run_name is None:
            summary_writer = tf.summary.FileWriter('{0}{1}'.format(self.summary_path, int(time.time())), graph=self.sess.graph)
        else:
            summary_writer = tf.summary.FileWriter('{0}{1}'.format(self.summary_path, run_name), graph=self.sess.graph)

        for epoch in range(num_epochs):
            if epoch > 0 and epoch % summary_interval == 0:
                if verbose:
                    print('epoch', epoch)
            for episode in range(batch_size):
                env.reset()
                while env.get_reward() is None:
                    # get legal moves
                    legal_moves = env.get_legal_moves()

                    # find best move and update trace sum
                    move_idx, _ = self.sess.run([self.next_board_idx,
                                                 self.train_op],
                                                feed_dict={self.turn_placeholder: env.turn,
                                                           self.board_placeholder: [env.board],
                                                           self.next_boards_placeholder: env.get_candidate_boards(),
                                                           self.reward_placeholder: 0.0})

                    move = legal_moves[move_idx]

                    # with probability epsilon:
                    # 1. ignore best move
                    # 2. make random move
                    # 3. reset traces
                    if np.random.rand() < epsilon:
                        move = choice(legal_moves)
                        self.sess.run(self.reset_traces_op)

                    # push the move onto the environment
                    env.make_move(move)

                # update traces with final state and reward
                self.sess.run([self.train_op,
                               self.average_turn_count_per_game_update_op],
                              feed_dict={self.turn_placeholder: env.turn,
                                         self.board_placeholder: [env.board],
                                         self.next_boards_placeholder: np.zeros((0, 3, 3, 3)),
                                         self.reward_placeholder: env.get_reward()})

                self.sess.run([self.reset_traces_op, self.reset_game_turn_count_op])
                env.reset()

            if epoch > 0 and epoch % summary_interval == 0:
                if verbose:
                    print('loss_ema:', self.sess.run(self.average_loss))

                self.test(env)

                self.saver.save(self.sess, self.checkpoint_path + 'checkpoint.ckpt')
                summary = self.sess.run(self.summaries_op, feed_dict={self.batch_size_placeholder: batch_size})
                summary_writer.add_summary(summary, (epoch+1)*batch_size)

            self.sess.run([self.reset_traces_op, self.reset_batch_turn_count_op])

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


