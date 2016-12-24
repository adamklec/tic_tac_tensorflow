import tensorflow as tf

from nn_model import NeuralNetworkModel
from game import TicTacToe


def main():
    with tf.Session() as sess:
        nn_model = NeuralNetworkModel(sess,
                                      '/Users/adam/Documents/projects/td_learning/tic_tac_toe/model/',
                                      '/Users/adam/Documents/projects/td_learning/tic_tac_toe/log/',
                                      '/Users/adam/Documents/projects/td_learning/tic_tac_toe/checkpoints/')
        env = TicTacToe()
        nn_model.train(env, 1000000, 100, 0.1, run_name='test2', verbose=True)

if __name__ == "__main__":
    main()
