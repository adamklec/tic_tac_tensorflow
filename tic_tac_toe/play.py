import tensorflow as tf
from human_agent import HumanAgent
from random_agent import RandomAgent
from nn_agent import NeuralNetworkAgent
from nn_agent import NeuralNetworkAgent
from game import TicTacToe


def main():
    with tf.Session() as sess:
        env = TicTacToe()
        nn_agent = NeuralNetworkAgent(sess,
                                      '',
                                      '',
                                      '/Users/adam/Documents/projects/td_learning/tic_tac_toe/checkpoints/',
                                      restore=True)
        # players = [HumanAgent(), NeuralNetworkAgent(nn_model)]
        players = [RandomAgent(), RandomAgent()]

        env.play(players, verbose=True)

if __name__ == "__main__":
    main()
