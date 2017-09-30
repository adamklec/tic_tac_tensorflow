import tensorflow as tf

from agents.backward_view_agent import BackwardViewAgent
from envs.tic_tac_toe import TicTacToeEnv
from value_model import ValueModel
import time


def main():
    model = ValueModel()
    env = TicTacToeEnv()
    nn_agent = BackwardViewAgent('agent_0', model, env)
    log_dir = "/Users/adam/Documents/projects/td_tic_tac_toe/log/" + str(int(time.time()))
    summary_op = tf.summary.merge_all()
    scaffold = tf.train.Scaffold(summary_op=summary_op)
    increment_global_step = tf.assign_add(nn_agent.global_episode_count, 1)
    with tf.train.MonitoredTrainingSession(checkpoint_dir=log_dir,
                                           scaffold=scaffold) as sess:
        nn_agent.sess = sess
        env.sess = sess

        while True:
            episode_idx = sess.run(nn_agent.global_episode_count)

            if episode_idx % 1000 == 0:
                results = env.random_agent_test(nn_agent.get_move_function())
                print(episode_idx, ':', results)
                sess.run(increment_global_step)
            else:
                reward = nn_agent.train(0.1)
                # print("%i: %i" % (episode_idx, reward))

if __name__ == "__main__":
    main()
