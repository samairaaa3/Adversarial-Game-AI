# Adversarial Game AI (Minimax, Alpha-Beta, MCTS)

This project implements adversarial game-playing agents using classical search and simulation-based methods. The agents make decisions in a turn-based environment using **Minimax**, **Alpha-Beta Pruning**, and **Monte Carlo Tree Search (MCTS)** to choose actions that maximize expected outcome against an opponent.

## Features

- **Minimax Agent:** Explores the game tree assuming an optimal opponent and selects actions that maximize worst-case reward.  
- **Alpha-Beta Agent:** Optimizes Minimax search by pruning branches that cannot affect the final decision, improving efficiency.  
- **Monte Carlo Tree Search Agent:** Uses random rollouts and visit statistics to approximate action values without full tree expansion.  
- **Strategy Comparison:** Enables side-by-side evaluation of different agents in the same game environment.

## Techniques Used

- Adversarial Search  
- Minimax and Alpha-Beta Pruning  
- Monte Carlo Tree Search (MCTS)  
- Tree-based state evaluation  
- Simulation and rollout-based value estimation

## Included Files

- `games.py` — Core game and agent framework, including adversarial search logic.  
- `monteCarlo.py` — Implementation of the Monte Carlo Tree Search agent.  
- `util.py` — Shared utility functions used by the agents (if included).  
- `game.py` — Game definitions and interfaces (if included).

## How to Run 

This project is designed to plug into a turn-based game environment (for example, a grid-based or Pacman-style game). A typical run command in a compatible environment might look like:

```bash
python your_game_driver.py -p MinimaxAgent
python your_game_driver.py -p AlphaBetaAgent
python your_game_driver.py -p MonteCarloAgent
