class HumanAgent(object):
    def __init__(self, env):
        self.env = env

    def get_move(self):
        while True:
            legal_moves = self.env.get_legal_moves()
            move = input("Enter your move:")
            try:
                move = int(move) - 1
                if move in legal_moves:
                    return move
            except ValueError:
                print("Illegal move")
