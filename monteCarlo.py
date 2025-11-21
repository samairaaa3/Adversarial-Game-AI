
import copy
import random
import time
import sys
import math
from collections import namedtuple

GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

# MonteCarlo Tree Search support
# Total 40 pt for monteCarlo.py
class MCTS:
    class Node:
        def __init__(self, state, par=None):
            self.state = copy.deepcopy(state)

            self.parent = par
            self.children = []
            self.visitCount = 0
            self.winScore = 0

        def getChildWithMaxScore(self):
            if not self.children:
                return None
            return max(self.children, key=lambda x: x.visitCount)

    def __init__(self, game, state):
        self.root = self.Node(state)
        self.state = state
        self.game = game
        self.exploreFactor = math.sqrt(2)

    def monteCarloPlayer(self, timelimit=4):
        """Entry point for Monte Carlo tree search"""
        start = time.perf_counter()
        end = start + timelimit
        """
        Use time.perf_counter() to apply iterative deepening strategy.
         At each iteration we perform 4 stages of MCTS: 
         SELECT, 
         EXPAND, 
         SIMULATE, 
         and BACKUP. 

         Once time is up use getChildWithMaxScore() to pick the node to move to
        """
        while time.perf_counter() < end:
            promisingNode = self.selectNode(self.root)
            if not self.game.terminal_test(promisingNode.state):
                self.expandNode(promisingNode)
            nodeToSimulate = promisingNode
            if promisingNode.children:
                nodeToSimulate = random.choice(promisingNode.children)
            playoutResult = self.simulateRandomPlay(nodeToSimulate)
            self.backPropagation(nodeToSimulate, playoutResult)

        winnerNode = self.root.getChildWithMaxScore()
        assert (winnerNode is not None)
        return winnerNode.state.move


    """SELECT stage function. walks down the tree using findBestNodeWithUCT()"""

    def selectNode(self, nd):
        node = nd
        while node.children:
            node = self.findBestNodeWithUCT(node)
        return node

    def findBestNodeWithUCT(self, nd):
        """finds the child node with the highest UCT. 
        Parse nd's children and use uctValue() to collect ucts 
        for the children.....
        Make sure to handle the case when uct value of 2 or more children
        nodes are the same."""
        childUCT = []
        for child in nd.children:
            uct = self.uctValue(nd.visitCount, child.winScore, child.visitCount)
            childUCT.append((uct, child))

        maxUCT = max(childUCT, key=lambda x: x[0])[0]
        bestChildren = [child for uct, child in childUCT if uct == maxUCT]

        return random.choice(bestChildren)

    def uctValue(self, parentVisit, nodeScore, nodeVisit):
        """compute Upper Confidence Value for a node"""
        if nodeVisit == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        return (nodeScore / nodeVisit) + self.exploreFactor * math.sqrt(math.log(parentVisit) / nodeVisit)

    def expandNode(self, nd):
        """generate the child nodes for node nd. For convenience, generate
        all the child nodes and attach them to nd."""
        stat = nd.state
        for move in self.game.actions(stat):
            newState = self.game.result(stat, move)
            childNode = self.Node(newState, nd)
            nd.children.append(childNode)

    """SIMULATE stage function"""

    def simulateRandomPlay(self, nd):
        """
        This function retuns the result of simulating off of node nd a 
        termination node, and returns the winner 'X' or 'O' or 0 if tie.
        Note: pay attention nd may be itself a termination node. Use compute_utility 
        to check for it.
        """
        state = nd.state
        current = state

        while not self.game.terminal_test(current):
            actions = self.game.actions(current)
            move = random.choice(actions)
            current = self.game.result(current, move)

        utility = self.game.compute_utility(current.board, self.game.to_move(state))

        if utility > 0:
            return 'X'
        elif utility < 0:
            return 'O'
        else:
            return 0

    def backPropagation(self, nd, winningPlayer):
        """propagate upword to update score and visit count from
        the current leaf node to the root node."""
        node = nd
        while node is not None:
            node.visitCount += 1
            if self.game.to_move(node.state) != winningPlayer and winningPlayer != 0:
                node.winScore += 1
            node = node.parent


def monte_carlo_player(game, state):
    mcts = MCTS(game, state)
    return mcts.monteCarloPlayer(timelimit=game.timer if game.timer > 0 else 4)
