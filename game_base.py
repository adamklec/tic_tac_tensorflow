class Game:

    @staticmethod
    def new():
        game = Game()
        game.reset()
        return game

    def extract_features(self):
        raise NotImplementedError("Please Implement this method")

    def play(self, players):
        player_num = 0
        while self.reward() is None:
            self.next_step(players[player_num])
            player_num = (player_num + 1) % 2
        return self.reward()

    def next_step(self, player, player_num, draw=False):
        raise NotImplementedError("Please Implement this method")

    def clone(self):
        """
        Return an exact copy of the game. Changes can be made
        to the cloned version without affecting the original.
        """
        raise NotImplementedError("Please Implement this method")

    def get_legal_moves(self):
        raise NotImplementedError("Please Implement this method")

    def reset(self):
        """
        Resets game to original layout.
        """
        raise NotImplementedError("Please Implement this method")

    def reward(self):
        """
        Get reward.
        """
        raise NotImplementedError("Please Implement this method")

    def is_valid_move(self, action):
        raise NotImplementedError("Please Implement this method")
