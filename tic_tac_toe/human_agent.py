class HumanAgent(object):
    def get_move(self, env):
        while True:
            move = int(input("Enter your move [1-9]")) - 1
            if move in env.get_legal_moves():
                return move
            else:
                print("Illegal move")
