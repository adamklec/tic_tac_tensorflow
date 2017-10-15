import tensorflow as tf
from agents.human_agent import HumanAgent
from agents.backward_agent import BackwardAgent
from agents.forward_agent import ForwardAgent
from model import ValueModel
from env import TicTacToeEnv


def main():
    log_dir = '/Users/adam/Documents/projects/td_tic_tac_toe/log/forward'
    env = TicTacToeEnv()
    model = ValueModel(env.feature_vector_size, 1000)
    # agent = BackwardViewAgent('agent_0', model, env)
    agent = ForwardAgent('agent_0', model, env)
    human = HumanAgent()

    with tf.train.SingularMonitoredSession(checkpoint_dir=log_dir) as sess:
        agent.sess = sess
        env.sess = sess
        players = [agent, human]
        env.play(players, verbose=True)

if __name__ == "__main__":
    main()
