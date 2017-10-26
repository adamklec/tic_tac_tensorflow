import tensorflow as tf
from agents.simple_agent import SimpleAgent
from agents.td_agent import TDAgent
from agents.forward_agent import ForwardAgent
from agents.backward_agent import BackwardAgent
from agents.leaf_agent import LeafAgent
from agents.random_agent import RandomAgent
from env import TicTacToeEnv
from model import ValueModel


def main():
    env = TicTacToeEnv()
    model = ValueModel(env.feature_vector_size, 100)

    # agent = SimpleAgent('agent_0', model, env)
    # agent = TDAgent('agent_0', model, env)
    agent = ForwardAgent('agent_0', model, env)
    # agent = BackwardAgent('agent_0', model, env)
    # agent = LeafAgent('agent_0', model, env)

    random_agent = RandomAgent(env)

    log_dir = "./log/forward3"

    summary_op = tf.summary.merge_all()
    scaffold = tf.train.Scaffold(summary_op=summary_op)
    with tf.train.MonitoredTrainingSession(checkpoint_dir=log_dir,
                                           scaffold=scaffold) as sess:
        agent.sess = sess
        env.sess = sess

        next_test_idx = 0

        while True:
            step_count = sess.run(agent.global_step_count)
            if step_count >= next_test_idx:
                results = random_agent.test(agent)

                sess.run(agent.update_random_agent_test_results,
                         feed_dict={random_agent_test_: result
                                    for random_agent_test_, result in zip(agent.random_agent_test_s, results)})
                next_test_idx = next_test_idx + 10000
                sess.run(agent.increment_global_step_count)
                print(step_count, ':', results)
                if results[2] + results[5] == 0:
                    sess.run(summary_op)
                    break
            else:
                agent.train(.2)

if __name__ == "__main__":
    main()
