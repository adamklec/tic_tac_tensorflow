import tensorflow as tf
from agents.human_agent import HumanAgent
from agents.simple_agent import SimpleAgent
from agents.td_agent import TDAgent
from agents.forward_agent import ForwardAgent
from agents.backward_agent import BackwardAgent
from agents.leaf_agent import LeafAgent
from model import ValueModel
from env import TicTacToeEnv


def main():
    log_dir = '/Users/adam/Documents/projects/td_tic_tac_toe/log/leaf2'
    env = TicTacToeEnv()
    model = ValueModel(env.feature_vector_size, 100)
    # agent = SimpleAgent('agent_0', model, env)
    # agent = TDAgent('agent_0', model, env)
    # agent = ForwardAgent('agent_0', model, env)
    # agent = BackwardAgent('agent_0', model, env)
    agent = LeafAgent('agent_0', model, env)
    human = HumanAgent(env)

    with tf.train.SingularMonitoredSession(checkpoint_dir=log_dir) as sess:
        agent.sess = sess
        env.sess = sess
        players = [human, agent]
        env.play(players, verbose=True)

if __name__ == "__main__":
    main()
