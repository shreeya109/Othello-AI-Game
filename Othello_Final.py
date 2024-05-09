import pygame
import numpy as np
import time
import copy 
import random

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 155, 0)
RED = (255, 0, 0)

# Standard values to be used
BOARD_SIZE = 8
SQUARE_SIZE = 50
SCREEN_SIZE = (BOARD_SIZE * SQUARE_SIZE, BOARD_SIZE * SQUARE_SIZE)

pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Othello")
font = pygame.font.Font(None, 36)

# Initiaizing the board with the starting positons 
# 1 = cur_player
# -1 = AI Agent
def init_board():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int) # board instance 
    start_position = BOARD_SIZE // 2 - 1
    board[start_position][start_position] = 1
    board[start_position + 1][start_position + 1] = 1
    board[start_position][start_position + 1] = -1
    board[start_position + 1][start_position] = -1
    return board

# All the 8 directions you can make a move in 
dir = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# This method computes all valid moves given the current board configuration and cur_player
def valid_moves(board, cur_player):
    valid_m = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            # checking that there is no piece cuurrently at that position and if some disk is getting flipped by that move in any direction
            if board[y][x] == 0 and any(fip_check(board, x, y, dir_x, dir_y, cur_player) for dir_x, dir_y in dir):
                valid_m.append((x, y))
    return valid_m

# This method checks if a disk can be flipped as a result of a move
# dir_x - along x axis
# dir_y - along y axis
def fip_check(board, x, y, dir_x, dir_y, cur_player):
    x += dir_x
    y += dir_y
    # checks for the limits of the board
    if x >= BOARD_SIZE or x < 0 or y >= BOARD_SIZE or y < 0 or board[y][x] != -cur_player:
        return False
    # move further along the direction
    x += dir_x
    y += dir_y
    while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
        # checks if the opponents coins are between two of the cur_player's coins
        if board[y][x] == cur_player:
            return True
        elif board[y][x] == 0:
            break
        x += dir_x
        y += dir_y
    return False

# This method attempts to make a move for a cur_player given the current board configuration, attempted move, and teh cur_player
def make_move(board, move, cur_player):
    x, y = move
    if board[y][x] == 0: # if empty
        dir = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        flip = []
        for dir_x, dir_y in dir:
            # if a disk can be flipped as a result of the move then the direction of that disk is added to the list of disks to be flipped
            if fip_check(board, x, y, dir_x, dir_y, cur_player):
                flip.append((dir_x, dir_y))
        if flip: # not empty
            board[y][x] = cur_player
            # flips all the disks of the opponent as a result of the move made by the cur_player
            for dir_x, dir_y in flip:
                x_i, y_i = x + dir_x, y + dir_y
                while board[y_i][x_i] == -cur_player:
                    board[y_i][x_i] = cur_player
                    x_i += dir_x
                    y_i += dir_y
    return board

# This is GUI - draws the board given the current board configuration and last move played by a cur_player
def draw_board(board, last_move_made=None):
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            background = (x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, GREEN, background)
            pygame.draw.rect(screen, BLACK, background, 1)
            if last_move_made and (x, y) == last_move_made:
                pygame.draw.rect(screen, RED, background) # higlights the last move made by the current cur_player for conveinience
            if board[y][x] != 0:
                color = BLACK if board[y][x] == 1 else WHITE
                pygame.draw.circle(screen, color, (x * SQUARE_SIZE + SQUARE_SIZE // 2, y * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 4)

# Calculates the best possible move for the AI cur_player using the minimax algorithm along with alpha-beta pruning
def ai_move(board, cur_player):
    top_score = float('-inf') # worst case 
    best_move = None # worst case
    alpha = float('-inf')
    beta = float('inf')
    for move in valid_moves(board, cur_player): # all valid moves
        board_copy = make_move(np.copy(board), move, cur_player) # state of the board after applying that particular move
        score = minimax(board_copy, depth=5, cur_player=-cur_player, alpha=alpha, beta=beta) # difficulty of AI agent can be changed by increasing the depth
        if score > top_score: # finds the best possible score and move
            top_score = score
            best_move = move
        alpha = max(alpha, top_score)
    return best_move # best posisble move is returned

# The recursive minimax function that evaluates the board score
def minimax(board, depth, cur_player, alpha, beta):
    if depth == 0: # Board condition is evaluated when a leaf node is reached 
        return evaluate_board(board) 
    moves = valid_moves(board, cur_player)
    if not moves: # If there are no valid moves possible then board condition is evaluated
        return evaluate_board(board)
    if cur_player == 1:
        maximum_eval = float('-inf')
        for move in moves: # exploring a move out of all valid moves
            board_copy = make_move(np.copy(board), move, cur_player)
            eval = minimax(board_copy, depth - 1, -cur_player, alpha, beta) # recursive call to minimax
            maximum_eval = max(maximum_eval, eval) # maximum value obtained so far
            alpha = max(alpha, eval)
            if beta <= alpha:
                break 
        return maximum_eval
    else:
        minimum_eval = float('inf')
        for move in moves: # exploring a move out of all valid moves
            board_copy = make_move(np.copy(board), move, cur_player) # recursive call to minimax
            eval = minimax(board_copy, depth - 1, -cur_player, alpha, beta)
            minimum_eval = min(minimum_eval, eval) # maximum value obtained so far
            beta = min(beta, eval)
            if beta <= alpha:
                break 
        return minimum_eval

# Evaluates the best possible score given the current board configuration
def evaluate_board(board):
    empty_squares = np.sum(board == 0)
    total_squares = BOARD_SIZE * BOARD_SIZE
    stage_game = empty_squares / total_squares

    # changing heurestics based on the number of empty squares in the grid.
    if stage_game > 0.75: # early stage of the game
        mobility_weight = 10 # early on the weightage of having more moves possible is more as it can help cur_player being fourced into unvaourable positions
        parity_weight = 5 #early on coins keep flipping so having more coins does not have that much weightage
        corner_weight = 100
        corner_closeness_weight = 100
        stability_weight = 15
    elif stage_game > 0.25: # middle stage of the game
        mobility_weight = 5
        parity_weight = 10
        corner_weight = 80
        corner_closeness_weight = 80
        stability_weight = 20
    else: # end stage of the game
        mobility_weight = 1
        parity_weight = 25 
        corner_weight = 50 # as the end of the game approaches occupying the corners is more valuable
        corner_closeness_weight = 50 # as the end of the game approaches occupying the cells near the corner is more valuable
        stability_weight = 30 # as the end of the game approaches stable disks become more valuable

    scores = {
        1: 0, 
        -1: 0
    }

    # number of valid moves 
    cur_player_mobility = len(valid_moves(board, -1))
    opponent_mobility = len(valid_moves(board, 1))
    scores[1] += (cur_player_mobility-opponent_mobility) * mobility_weight
    
    # number of disks of a cur_player on a board 
    cur_player_coins = np.sum(board == -1)
    opponent_coins = np.sum(board == 1)
    scores[1] += (cur_player_coins - opponent_coins) * parity_weight

    # number of corners occupied 
    corners = [(0, 0), (0, BOARD_SIZE-1), (BOARD_SIZE-1, 0), (BOARD_SIZE-1, BOARD_SIZE-1)]
    cur_player_corners = sum([board[y, x] == -1 for x, y in corners])
    opponent_corners = sum([board[y, x] == 1 for x, y in corners])
    scores[1] += (cur_player_corners - opponent_corners) * corner_weight

    # closenesss to the corner.
    cur_player_corner_closeness = corner_closeness(board, -1)
    opponent_corner_closeness = corner_closeness(board, 1)
    scores[1] += (opponent_corner_closeness - cur_player_corner_closeness) * corner_closeness_weight

    # The number of discs that cannot be flipped ireespective of any move
    cur_player_stability = sum([1 for disc in stable_discs_board(board, -1)])
    opponent_stability = sum([1 for disc in stable_discs_board(board, 1)])
    scores[1] += (cur_player_stability - opponent_stability) * stability_weight
    
    return scores[1]

# Function to check closeness to corner 
def corner_closeness(board, cur_player):
    closeness_val = 0
    # Checking cells directly adjacent to the corner
    corners = {
        (0, 0): [(0, 1), (1, 0), (1, 1)],
        (0, BOARD_SIZE - 1): [(0, BOARD_SIZE - 2), (1, BOARD_SIZE - 1), (1, BOARD_SIZE - 2)],
        (BOARD_SIZE - 1, 0): [(BOARD_SIZE - 2, 0), (BOARD_SIZE - 1, 1), (BOARD_SIZE - 2, 1)],
        (BOARD_SIZE - 1, BOARD_SIZE - 1): [(BOARD_SIZE - 2, BOARD_SIZE - 1), (BOARD_SIZE - 1, BOARD_SIZE - 2), (BOARD_SIZE - 2, BOARD_SIZE - 2)]
    }

    #Checking corner and adjacent positions
    for corner, adjacent_positions in corners.items():
        corner_x, corner_y = corner
        if board[corner_y, corner_x] == 0 or board[corner_y, corner_x] == cur_player:
            for pos in adjacent_positions:
                pos_x, pos_y = pos
                #If the cur_player has an adjacent position he is rewarded 
                if board[pos_y, pos_x] == cur_player:
                    closeness_val += 1
                #If the opponent has an adjacent position he is punished 
                elif board[pos_y, pos_x] == -cur_player:
                    closeness_val -= 1

    return closeness_val

# Claculate number of dics that cannot be flipped by any move
def stable_discs_board(board, cur_player):
    stable_discs = set()
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == cur_player:
                #Checking if a position on the board is a stable disc
                if is_stable_disc(board, x, y, cur_player):
                    stable_discs.add((x, y))
    return stable_discs

#Checking if a position on the board is a stable disc
def is_stable_disc(board, x, y, cur_player):
    dir = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dir_x, dir_y in dir:
        # Checking if anything can be flipped in the direction
        if not is_stable_direction_direction(board, x, y, dir_x, dir_y, cur_player):
            return False
    return True

 # Checking if anything can be flipped in the dir
def is_stable_direction_direction(board, x, y, dir_x, dir_y, cur_player):
    opponent = -cur_player
    # Going along that particular direction
    x += dir_x
    y += dir_y
    # Checking along a given direction and no disc in that direction belongs to the opponent
    while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
        if board[y][x] == opponent:
            return False 
        elif board[y][x] == cur_player:
            x += dir_x
            y += dir_y
        else:
            return True  
    return True 

# End condition where nobody has valid moves
def game_over(board):
    return not valid_moves(board, 1) and not valid_moves(board, -1)

# Calculating the final socre by adding up number of discs occupied by each cur_player
def display_final_score(board):
    cur_player_score = np.sum(board == 1)
    opponent_score = np.sum(board == -1)
    
    message = f'You: {cur_player_score}, AI: {opponent_score}'
    text = font.render(message, True, WHITE)
    
    screen.fill(BLACK)
    screen.blit(text, (SCREEN_SIZE[0] // 2 - text.get_width() // 2, SCREEN_SIZE[1] // 2 - text.get_height() // 2))
    pygame.display.flip()
    pygame.time.wait(10000) 


def play_game():
    board = init_board()
    cur_player = 1
    game_running = True
    clock = pygame.time.Clock()
    ai_turn = False
    last_move_made = None

    while game_running:
        # end condition
        if game_over(board):
            display_final_score(board)
            game_running = False
            continue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not ai_turn:
                # Getting square from cursor
                x_axis_mouse, y_axis_mouse = event.pos
                x, y = x_axis_mouse // SQUARE_SIZE, y_axis_mouse // SQUARE_SIZE
                if (x, y) in valid_moves(board, cur_player):
                    board = make_move(board, (x, y), cur_player)
                    last_move_made = (x, y)
                    draw_board(board, last_move_made)  
                    pygame.display.flip()
                    pygame.time.wait(3000) 
                    ai_turn = True

        if ai_turn:
            ai_move_pos = ai_move(board, -cur_player)
            if ai_move_pos:
                board = make_move(board, ai_move_pos, -cur_player)
                last_move_made = ai_move_pos
            ai_turn = False

        draw_board(board, last_move_made)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def random_cur_player(board, cur_player):
    moves = valid_moves(board, cur_player)
    if moves:
        return random.choice(moves)
    return None

def simulate_gmaes(num_games, display_results=True):
    # Store results of computation of running games
    results = {'AI': 0, 'random': 0, 'draw': 0}
    result_setails = []
    
    for game_number in range(num_games):
        board = init_board()
        cur_player = 1
        
        while not game_over(board):
            if cur_player == 1:
                move = ai_move(board, 1)
            else:
                move = random_cur_player(board, -1)
            
            if move:
                board = make_move(board, move, cur_player)
            cur_player = -cur_player
        
        # Computing the final score expternally
        ai_score = np.sum(board == 1)
        human_random_score = np.sum(board == -1)

        # Find out who won
        if ai_score > human_random_score:
            results['AI'] += 1
            outcome = "AI won"
        elif ai_score < human_random_score:
            results['random'] += 1
            outcome = "Random wins"
        else:
            results['draw'] += 1
            outcome = "Draw"

        if display_results:
            print(f"Game {game_number + 1}: {outcome}")
            print(f"AI score: {ai_score}")
            print(f"Random score: {human_random_score}")

    return results, result_setails



def test():

    num_games = 10  # Number of games to run
    results = simulate_gmaes(num_games, display_results=True)
    total_games = results['AI'] + results['random'] + results['draw']
    win_rate = results['AI'] / total_games   
    print("Win Rate: ", win_rate)
        
        
        
        

    


def main():
    #Either testing or trainng
    choice = input("Would you like to 'test' or 'play'? ").strip().lower()

    if choice == "test":
        test()
    elif choice == "play":
        play_game() 
    else:
        print("Invalid choice. Please enter 'train' or 'play'.")
        main() 




if __name__ == "__main__":
    main()