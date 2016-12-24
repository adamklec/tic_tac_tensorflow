import tensorflow as tf

from model import Ne
from game import TicTacToe


def main():
    with tf.Session() as sess:
        env = TicTacToe()
        nn_agent = NeuralNetAgent(env,
                                  sess,
                                  '/Users/adam/Documents/projects/td_learning/tic_tac_toe/model/',
                                  '/Users/adam/Documents/projects/td_learning/tic_tac_toe/log/',
                                  '/Users/adam/Documents/projects/td_learning/tic_tac_toe/checkpoints/')
        nn_agent.train(1000000, 100, 0.1, run_name='boltzmann', verbose=True)

if __name__ == "__main__":
    main()
