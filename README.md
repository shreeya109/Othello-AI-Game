# Othello AI Game

## Description

This project is a Python implementation of the classic board game Othello (also known as Reversi) featuring an AI opponent. The game is rendered graphically using the Pygame library. Players take turns placing disks on the board with the aim to trap and flip their opponent's disks. The AI uses a minimax algorithm with alpha-beta pruning to determine its moves, offering a challenging experience for players.

## Installation

### Prerequisites

- Python 3.x
- Pygame
- NumPy


### Setup

Install Pygame and NumPy. You can install these packages using pip. Run the following commands in your terminal:

   ```bash
   pip install pygame
   pip install numpy
   ```

### Running the game


   ```bash
    python othello_ai.py
```

### Gameplay

Starting the Game: You can choose between 'test' and 'play'. Choosing 'test' will simulate games between the AI and a random player strategy. Choosing 'play' will start a game against the AI.

Making Moves: Click on the board to place your disk on an available spot during your turn.

AI Moves: After your move, the AI calculates its best move and places its disk automatically.

End Game: The game ends when no valid moves are available. The final score is then displayed.

Note: After entering 'play' the board appears in a pop window. If the program is running too slow then lower the depth in the metho ai_move. We have also attached a video demo of the gameplay.

