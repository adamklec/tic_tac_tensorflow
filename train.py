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
    # agent = ForwardAgent('agent_0', model, env)
    # agent = BackwardAgent('agent_0', model, env)
    agent = LeafAgent('agent_0', model, env)

    random_agent = RandomAgent(env)

    log_dir = "./log/leaf"

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir)

    scaffold = tf.train.Scaffold(summary_op=summary_op)
    with tf.train.MonitoredTrainingSession(checkpoint_dir=log_dir,
                                           scaffold=scaffold) as sess:
        agent.sess = sess
        env.sess = sess

        while True:
            episode_count = sess.run(agent.episode_count)
            if episode_count % 1000 == 0:
                results = random_agent.test(agent)

                sess.run(agent.update_random_agent_test_results,
                         feed_dict={random_agent_test_: result
                                    for random_agent_test_, result in zip(agent.random_agent_test_s, results)})
                print(episode_count, ':', results)

                if results[2] + results[5] == 0:
                    final_summary = sess.run(summary_op)
                    summary_writer.add_summary(final_summary, global_step=episode_count)
                    break
            else:
                agent.train(.2)
            sess.run(agent.increment_episode_count)


if __name__ == "__main__":
    main()
