"""Games or Adversarial Search (Chapter 5)"""

import copy
import random
import time
from collections import namedtuple

import numpy as np

# Total 60 pt for this script
# namedtuple used to generate game state:
GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def gen_state(move = '(1, 1)', to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """
        move = the move that has lead to this state,
        to_move=Whose turn is to move
        x_position=positions on board occupied by X player,
        o_position=positions on board occupied by O player,
        (optionally) number of rows, columns and how many consecutive X's or O's required to win,
    """
    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, move=move, utility=0, board=board, moves=moves)

# ______________________________________________________________________________
# MinMax Search
def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. If there are multiple
    choices for best move, then randomly choose one.
    Students to do: Add support to this function for state caching, so
    to avoid re-evaluating the same state more than once and save on memory.
    Important: Caching should be added to all search algorithms where ever it
    makes sense. without cache support you won't be able to reach the 
    time limit for various tests."""

    player = game.to_move(state)
    cache = {}

    def max_value(state):
        key = frozenset(state.board.items())
        if key in cache:
            return cache[key]
        if game.terminal_test(state):
            val = game.utility(state, player)
        else:
            val = -np.inf
            for a in game.actions(state):
                val = max(val, min_value(game.result(state, a)))
        cache[key] = val
        return val

    def min_value(state):
        key = frozenset(state.board.items())
        if key in cache:
            return cache[key]
        if game.terminal_test(state):
            val = game.utility(state, player)
        else:
            val = np.inf
            for a in game.actions(state):
                val = min(val, max_value(game.result(state, a)))
        cache[key] = val
        return val

    # Body of minmax:
    # to be implemented by students
    best_score = -np.inf
    best_actions = []
    for action in game.actions(state):
        value = min_value(game.result(state, action))
        if value > best_score:
            best_score = value
            best_actions = [action]
        elif value == best_score:
            best_actions.append(action)
    return random.choice(best_actions)


def minmax_cutoff(game, state):
    """MinMax with cutoff using eval1() at specified depth."""
    player = game.to_move(state)
    cache = {}

    def canonical(state):
        return (tuple(sorted(state.board.items())), state.to_move, len(state.moves))

    def max_value(state, depth):
        key = (canonical(state), depth)
        if key in cache:
            return cache[key]
        if game.terminal_test(state):
            val = game.utility(state, player)
        elif depth >= game.d:
            val = game.eval1(state)
        else:
            val = -float('inf')
            for a in game.actions(state):
                result = min_value(game.result(state, a), depth + 1)
                if result is None:
                    continue
                val = max(val, result)
        cache[key] = val
        return val

    def min_value(state, depth):
        key = (canonical(state), depth)
        if key in cache:
            return cache[key]
        if game.terminal_test(state):
            val = game.utility(state, player)
        elif depth >= game.d:
            val = game.eval1(state)
        else:
            val = float('inf')
            for a in game.actions(state):
                result = max_value(game.result(state, a), depth + 1)
                if result is None:
                    continue
                val = min(val, result)
        cache[key] = val
        return val

    best_score = -float('inf')
    best_actions = []

    for action in game.actions(state):
        value = min_value(game.result(state, action), 0)
        if value is None:
            continue
        if value > best_score:
            best_score = value
            best_actions = [action]
        elif value == best_score:
            best_actions.append(action)

    return random.choice(best_actions) if best_actions else random.choice(game.actions(state))
# ______________________________________________________________________________
def alpha_beta(game, state):
    """Search game to determine best action; use alpha-beta pruning.
     this version searches all the way to the leaves."""
    player = game.to_move(state)
    cache = {}
    def max_value(state, alpha, beta):
        key = frozenset(state.board.items())
        if key in cache:
            return cache[key]
        if game.terminal_test(state):
            val = game.utility(state, player)
        else:
            val = -np.inf
            for a in game.actions(state):
                val = max(val, min_value(game.result(state, a), alpha, beta))
                if val >= beta:
                    break
                alpha = max(alpha, val)
        cache[key] = val
        return val

    def min_value(state, alpha, beta):
        key = frozenset(state.board.items())
        if key in cache:
            return cache[key]
        if game.terminal_test(state):
            val = game.utility(state, player)
        else:
            val = np.inf
            for a in game.actions(state):
                val = min(val, max_value(game.result(state, a), alpha, beta))
                if val <= alpha:
                    break
                beta = min(beta, val)
        cache[key] = val
        return val

    alpha = -np.inf
    beta = np.inf
    best_score = -np.inf
    best_actions = []

    for action in game.actions(state):
        value = min_value(game.result(state, action), alpha, beta)
        if value > best_score:
            best_score = value
            best_actions = [action]
        elif value == best_score:
            best_actions.append(action)
    return random.choice(best_actions)



def alpha_beta_cutoff(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    player = game.to_move(state)
    cache = {}

    def canonical(state):
        return (tuple(sorted(state.board.items())), state.to_move, len(state.moves))

    def max_value(state, alpha, beta, depth):
        key = (canonical(state), depth)
        if key in cache:
            return cache[key]
        if game.terminal_test(state):
            val = game.utility(state, player)
        elif depth >= game.d:
            val = game.eval1(state)
        else:
            val = -float('inf')
            for a in game.actions(state):
                val = max(val, min_value(game.result(state, a), alpha, beta, depth + 1))
                if val >= beta:
                    break
                alpha = max(alpha, val)
        cache[key] = val
        return val

    def min_value(state, alpha, beta, depth):
        key = (canonical(state), depth)
        if key in cache:
            return cache[key]
        if game.terminal_test(state):
            val = game.utility(state, player)
        elif depth >= game.d:
            val = game.eval1(state)
        else:
            val = float('inf')
            for a in game.actions(state):
                val = min(val, max_value(game.result(state, a), alpha, beta, depth + 1))
                if val <= alpha:
                    break
                beta = min(beta, val)
        cache[key] = val
        return val

    alpha = -float('inf')
    beta = float('inf')
    best_score = -float('inf')
    best_actions = []

    for action in game.actions(state):
        value = min_value(game.result(state, action), alpha, beta, 1)
        if value > best_score:
            best_score = value
            best_actions = [action]
        elif value == best_score:
            best_actions.append(action)

    return random.choice(best_actions)

def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    """uses alphaBeta prunning with minmax, or with cutoff version, for AI player"""
    
    """Use a method to speed up at the start to avoid search down a long tree with not much outcome.
    Hint: for speedup use random_player for start of the game when you see search time is too long"""
    if len(state.moves) == game.size * game.size:
        return random_player(game, state)

    if (game.timer < 0):
        game.d = -1
        return alpha_beta(game, state)

    start = time.perf_counter()
    end = start + game.timer
    """use the above timer to implement iterative deepening using alpha_beta_cutoff() version"""
    move = None
    game.d = 1
    while True:
        current_time = time.perf_counter()
        if current_time >= end:
            break
        try:
            next_move = alpha_beta_cutoff(game, state)
            move = next_move
            game.d += 1
        except Exception:
            break

    print("iterative deepening to depth: ", game.d - 1)
    return move if move else random_player(game, state)


def minmax_player(game, state):
    """Uses minmax or minmax with cutoff depth, for AI player.
    Use a method to speed up at the start to avoid search down a deep tree with not much outcome."""

    # Opening move shortcut: play randomly to avoid wasting time
    if len(state.moves) == game.size * game.size:
        return random_player(game, state)

    # Full-depth search if no time limit
    if game.timer < 0:
        game.d = -1
        return minmax(game, state)

    # Iterative deepening using timer-based cutoff
    game.d = 1
    move = random_player(game, state)  # fallback in case time runs out
    start = time.perf_counter()
    end = start + game.timer

    while time.perf_counter() < end:
        try:
            next_move = minmax_cutoff(game, state)
            move = next_move
            game.d += 1
        except Exception:
            break

    print("minmax_player: iterative deepening to depth:", game.d - 1)
    return move

# ______________________________________________________________________________
# base class for Games

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))

class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to_move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    depth = -1 means max search tree depth to be used."""

    def __init__(self, size=3, k=3, t=-1):
        self.size = size
        if k <= 0:
            self.k = size
        else:
            self.k = k
        self.d = -1 # d is cutoff depth. Default is -1 meaning no depth limit. It is controlled usually by timer
        self.maxDepth = size * size # max depth possible is width X height of the board
        self.timer = t #timer  in seconds for opponent's search time limit. -1 means unlimited
        moves = [(x, y) for x in range(1, size + 1)
                 for y in range(1, size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def reset(self):
        moves = [(x, y) for x in range(1, self.size + 1)
                 for y in range(1, self.size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    @staticmethod
    def switchPlayer(player):
        assert(player == 'X' or player == 'O')
        return 'O' if player == 'X' else 'X'

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        try:
            moves = list(state.moves)
            moves.remove(move)
        except (ValueError, IndexError, TypeError) as e:
            print("exception: ", e)

        return GameState(to_move=self.switchPlayer(state.to_move), move=move,
                         utility=self.compute_utility(board, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the state value to player; state.k for win, -state.k for loss, 0 otherwise. k is dimension of the board (3, 4, ...)"""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or lost or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(0, self.size):
            for y in range(1, self.size + 1):
                print(board.get((self.size - x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, player):
        """If player wins with this move, return k if player is 
        'X' and -k if 'O' else return 0. 
        For students to do: Use k_in_row for checking if there are k cells
        in line in a direction."""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for direction in directions:
            found, _ = self.k_in_row(board, player, direction, self.k, self.size)
            if found:
                return self.k if player == 'X' else -self.k
        return 0

    def k_in_row(self, board, player, direction, k, size):
        """Check if there are k in a row for player in board in given direction.
        direction is a pair like (0, 1), meaning check along rows.
        Returns (True/False, count of lines with exactly k)."""
        dx, dy = direction
        count = 0
        for x in range(1, size + 1):
            for y in range(1, size + 1):
                i = 0
                while i < k:
                    nx = x + dx * i
                    ny = y + dy * i
                    if not (1 <= nx <= size and 1 <= ny <= size):
                        break
                    if board.get((nx, ny)) != player:
                        break
                    i += 1
                if i == k:
                    count += 1
        return (count > 0, count)
        
    # evaluation function, version 1
    def eval1(self, state):
        """design and implement evaluation function for state.
        Here's an idea: 
        	 Use the number of k-1 or less matches for X and O For example you can fill up the following
        	 function possibleKComplete() which will use k_in_row(). This function
        	 would return number of say k-1 matches for a specific player, 'X', or 'O'.
        	 Anyhow, this is an idea. We have been able to use such method to get almost
        	 perfect playing experience.
        Again, remember we want this evaluation function to be represent
        how to good the state is for the player to win the game from here,
        and also it needs to be fast to compute.
        """
        board = state.board
        k = self.k
        n = self.size
        player = state.to_move
        opponent = 'O' if player == 'X' else 'X'
        score = 0

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        def count_k_minus_1_lines(player_symbol):
            count = 0
            for x in range(1, n + 1):
                for y in range(1, n + 1):
                    for dx, dy in directions:
                        current_line = []
                        for i in range(k):
                            nx = x + i * dx
                            ny = y + i * dy
                            if 1 <= nx <= n and 1 <= ny <= n:
                                current_line.append((nx, ny))
                            else:
                                break
                        if len(current_line) != k:
                            continue

                        player_marks = 0
                        opponent_marks = 0
                        empty = 0
                        for pos in current_line:
                            val = board.get(pos)
                            if val == player_symbol:
                                player_marks += 1
                            elif val == '.':
                                empty += 1
                            elif val is not None:
                                opponent_marks += 1

                        if player_marks == k:
                            count += 1000  # win
                        elif player_marks == k - 1 and opponent_marks == 0:
                            count += 10
                        elif player_marks == k - 2 and opponent_marks == 0:
                            count += 3
            return count

        score += count_k_minus_1_lines(player)
        score -= count_k_minus_1_lines(opponent)

        return score

