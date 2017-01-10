import tensorflow as tf
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.nn_agent import NeuralNetworkAgent
from tic_tac_toe.tic_tac_toe import TicTacToe


def main():
    with tf.Session() as sess:
        env = TicTacToe()
        nn_agent = NeuralNetworkAgent(sess, restore=True)
        players = [nn_agent, nn_agent]
        # players = [RandomAgent(), RandomAgent()]

        env.play(players, verbose=True)

if __name__ == "__main__":
    main()
