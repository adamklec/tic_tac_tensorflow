import tensorflow as tf

from agents.nn_agent import NeuralNetworkAgent
from tic_tac_toe import TicTacToe


def main():
    with tf.Session() as sess:
        nn_agent = NeuralNetworkAgent(sess)
        env = TicTacToe()
        nn_agent.train(env, 1000000, 100, 0.1, verbose=True)

if __name__ == "__main__":
    main()
