import tensorflow as tf
from agents.backward_view_agent import BackwardViewAgent
from agents.forward_view_agent import ForwardViewAgent
from env import TicTacToeEnv
from model import ValueModel


def main():
    env = TicTacToeEnv()
    model = ValueModel(env.feature_vector_size, 1000)

    # agent = BackwardViewAgent('agent_0', model, env)
    agent = ForwardViewAgent('agent_0', model, env)

    log_dir = "./log/forward"

    summary_op = tf.summary.merge_all()
    scaffold = tf.train.Scaffold(summary_op=summary_op)
    with tf.train.MonitoredTrainingSession(checkpoint_dir=log_dir,
                                           scaffold=scaffold) as sess:
        agent.sess = sess
        env.sess = sess

        next_test_idx = 0
        while True:
            step_count = sess.run(agent.global_step_count)
            if step_count >= next_test_idx or step_count == 0:
                results = env.random_agent_test(agent.get_move_function())
                next_test_idx = next_test_idx + 1000
                sess.run(agent.increment_global_step_count)
                print(step_count, ':', results)
            else:
                agent.train(0.1)


if __name__ == "__main__":
    main()
