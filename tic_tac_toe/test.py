from tic_tac_toe import TicTacToe
from nn_agent import NeuralNetAgent
from random_agent import RandomAgent
import tensorflow as tf


def main():
    with tf.Session() as sess:
        env = TicTacToe()
        nn_agent = NeuralNetAgent(env,
                                  sess,
                                  '/Users/adam/Documents/projects/chess_deep_learning/model/',
                                  '/Users/adam/Documents/projects/chess_deep_learning/log/',
                                  '/Users/adam/Documents/projects/chess_deep_learning/checkpoints/')
        nn_agent.restore()
        random_agent = RandomAgent()

        winner = env.play([nn_agent, random_agent])
        print(winner)


if __name__ == "__main__":
    main()
