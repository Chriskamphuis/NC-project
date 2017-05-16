import numpy as np

class Game:

    def __init__(self, board=None):
        
        # Define players
        self.playerOne = None
        self.playerTwo = None

        # Define the board
        self.board = board
        if board is None:
            self.board = np.zeros((6, 7), dtype=np.int8)
        (self.height, self.width) = self.board.shape
 
            
    # Function to add player to game (sign-in)
    def set_player(self, player):
        if self.playerOne == None:
            self.playerOne = player
        elif self.playerTwo == None:
            self.playerTwo = player
        else:
            print("Cannot have more than two players.\nPlayer with value {0} refused.".format(player.value))
            
    # Function to switch players (who goes first)
    def switch_players(self):
        temp = self.playerOne
        self.playerOne = self.playerTwo
        self.playerTwo = temp
        


    # Return instance of the board
    def get_board(self):
        return self.board.copy()
        
    # Function that returns a list of legal moves in the current board
    def get_legal_moves(self):
        legal_moves = list()
        for move in range(0, self.width):
            played = sum([1 for e in self.board[:, move] if e != 0])
            if (played < self.height):
                legal_moves.append(move)
        return legal_moves
  
    # Function to play a single game. Player that starts is given as argument
    def play_game(self, player):
        
        winner = 0
        while(winner == 0 and self.not_full()):
            move = player.get_move(self.get_legal_moves())
            
            winner = self.play_move(move, player.value)
            
            if (player == self.playerOne):
                player = self.playerTwo
            else:
                player = self.playerOne
        return winner
        
    # Check if the board is full, for draws
    def not_full(self):
        return not self.board.all()
  
    # Function to apply a move to the board
    def play_move(self, move, value):
        played = sum([1 for e in self.board[:, move] if e != 0])
        
        if played > self.height-1:
            raise ValueError('The column is already full')
        
        self.board[self.height-1-played, move] = value
        
        if (self.end_state(move, self.height-1-played, value)):
            return value
        else:
            return 0
    
    # Boolean function to check if the game is done
    def end_state(self, col, row, value):
        # Check for horizontal
        if (self.check_neighbour(col, row, value, 1, 0) + self.check_neighbour(col-1, row, value, -1, 0) >= 4):
            return True
            
        # Check for vertical   
        elif (self.check_neighbour(col, row, value, 0, 1) + self.check_neighbour(col, row-1, value, 0, -1) >= 4):
            return True
        
        # Check for upwards diagonal
        elif (self.check_neighbour(col, row, value, -1, -1) + self.check_neighbour(col+1, row+1, value, 1, 1) >= 4):
            return True
            
        # Check for downwards diagonal
        elif (self.check_neighbour(col, row, value, -1, 1) + self.check_neighbour(col+1, row-1, value, 1, -1) >= 4):
            return True
        
        # No connect-four found     
        return False    
    
    # Recursive function to check if neighbour in direction (dx,dy) has the same value    
    def check_neighbour(self, col, row, value, dx, dy):
        if (col < 0 or col >= self.width or row < 0 or row >= self.height):
            return 0
    
        elif (self.board[row, col] == value):
            return 1 + self.check_neighbour(col+dx, row+dy, value, dx, dy)
            
        else:
            return 0
            
    # Function that resets the board
    def reset_board(self):
        self.board = self.board * 0
