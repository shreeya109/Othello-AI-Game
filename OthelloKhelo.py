import pygame
import numpy as np

# Board colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 155, 0)
RED = (255, 0, 0) 

BOARD_SIZE = 8
SQUARE_SIZE = 50
SCREEN_SIZE = (BOARD_SIZE * SQUARE_SIZE, BOARD_SIZE * SQUARE_SIZE)

pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Othello")
font = pygame.font.Font(None, 36) 

# Board set up - initializes the starting positions
def init_board():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    start = BOARD_SIZE// 2 - 1
    board[start][start] = 1
    board[start + 1][start + 1] = 1
    board[start][start + 1]= -1
    board[start + 1][start]= -1
    return board

# all the directions that the player can place the disk in
directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# This method finds all valid moves a player can make given the current board configuration
def valid_moves_finder(board, player):
    valid_moves = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 0 and any(flip_checker(board, x, y, p_x, p_y, player) for p_x, p_y in directions):
                valid_moves.append((x, y))
    return valid_moves

# Checks if a certain move can flip the other player's disk or not
def flip_checker(board, x, y, p_x, p_y, player):
    x += p_x
    y += p_y

    if x >= BOARD_SIZE or x < 0 or y >= BOARD_SIZE or y < 0 or board[y][x] != -player:
        return False
    
    x += p_x
    y += p_y

    while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
        if board[y][x] == player:
            return True
        elif board[y][x] == 0:
            break
        x += p_x
        y += p_y

    return False

# Reflects the move made by the player
def player_move(board, move, player):
    x, y = move
    if board[y][x] == 0:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        flip_move = []
        for p_x, p_y in directions:
            if flip_checker(board, x, y, p_x, p_y, player):
                flip_move.append((p_x, p_y))

        if flip_move:
            board[y][x] = player
            for p_x, p_y in flip_move:
                x_i, y_i = x + p_x, y + p_y

                while board[y_i][x_i] == -player:
                    board[y_i][x_i] = player
                    x_i += p_x
                    y_i += p_y
    return board

# GUI - draws the board given the current board configuration
def draw_board(board, move=None):
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            rect = (x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, GREEN, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)
            if move and (x, y) == move:
                pygame.draw.rect(screen, RED, rect)  
            if board[y][x] != 0:
                color = BLACK if board[y][x] == 1 else WHITE
                pygame.draw.circle(screen, color, (x * SQUARE_SIZE + SQUARE_SIZE // 2, y * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 4)

# It gives a countdown before the AI's move
def countdown(seconds):
    for i in range(seconds, 0, -1):
        overlay = pygame.Surface(SCREEN_SIZE)
        overlay.set_alpha(128)  
        overlay.fill((0, 0, 0))  
        
        screen.blit(overlay, (0, 0))  

        text = font.render(str(i), True, WHITE)
        screen.blit(text, (SCREEN_SIZE[0] // 2 - text.get_width() // 2, SCREEN_SIZE[1] // 2 - text.get_height() // 2))

        pygame.display.flip()
        pygame.time.wait(1000)


# Immplement the Ai's move using the Minimax algoritm along with alpha beta pruning
def ai_move(board, player):
    countdown(3)

    max_score = -float('inf')
    best_move = None
    for move in valid_moves_finder(board, player):
        new_board = player_move(np.copy(board), move, player)
        score = minimax(new_board, depth=3, player=-player)
        if score > max_score:
            max_score = score
            best_move = move
    return best_move

# performs a minimax search along with alpha-beta pruning to evaluate the best move
def minimax(board, depth, player):
    if depth == 0:
        return np.sum(board) * player 
    moves = valid_moves_finder(board, player)
    if not moves:
        return np.sum(board) * player
    if player == 1:
        return max(minimax(player_move(np.copy(board), move, player), depth-1, -player) for move in moves)
    else:
        return min(minimax(player_move(np.copy(board), move, player), depth-1, -player) for move in moves)

def main():
    board = init_board()
    player = 1
    game = True
    clock = pygame.time.Clock()
    ai_turn = False
    prev = None

    while game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not ai_turn:
                mouse_x, mouse_y = event.pos
                x, y = mouse_x // SQUARE_SIZE, mouse_y // SQUARE_SIZE
                if (x, y) in valid_moves_finder(board, player):
                    board = player_move(board, (x, y), player)
                    prev = (x, y)
                    ai_turn = True

        if ai_turn:
            ai_move_pos = ai_move(board, -player)
            if ai_move_pos:
                board = player_move(board, ai_move_pos, -player)
                prev = ai_move_pos
            ai_turn = False

        draw_board(board, prev)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()