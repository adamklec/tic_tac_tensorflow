import tensorflow as tf

from agents.nn_agent import NeuralNetworkAgent
from tic_tac_toe.game import TicTacToe


def main():
    with tf.Session() as sess:
        nn_agent = NeuralNetworkAgent(sess,
                                      '/Users/adam/Documents/projects/td_learning/tic_tac_toe/model/',
                                      '/Users/adam/Documents/projects/td_learning/tic_tac_toe/log2/',
                                      '/Users/adam/Documents/projects/td_learning/tic_tac_toe/checkpoints/')
        env = TicTacToe()
        nn_agent.train(env, 1000000, 100, 0.1, verbose=True)

if __name__ == "__main__":
    main()
