class HumanAgent(object):
    def get_move(self, env):
        while True:
            legal_moves = env.get_legal_moves()
            move = input("Enter your move:")
            try:
                move = int(move) - 1
                if move in legal_moves:
                    return move
            except ValueError:
                print("Illegal move")
