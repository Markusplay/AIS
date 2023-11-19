from util import manhattanDistance
from game import Directions
import random, util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState


def scoreEvaluationFunction(currentGameState: GameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState) -> str:
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def max_value(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            v = float('-inf')
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                v = max(v, min_value(successor, 1, depth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, ghost_index, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            v = float('inf')
            for action in state.getLegalActions(ghost_index):
                successor = state.generateSuccessor(ghost_index, action)
                if ghost_index == state.getNumAgents() - 1:
                    v = min(v, max_value(successor, depth - 1, alpha, beta))
                else:
                    v = min(v, min_value(successor, ghost_index + 1, depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        legalMoves = gameState.getLegalActions()
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')

        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = min_value(successor, 1, self.depth, alpha, beta)
            if score > alpha:
                alpha = score
                bestAction = action

        return bestAction


######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Calculate the distance to the closest food
    foodDistances = [manhattanDistance(pacmanPos, food) for food in foodGrid.asList()]
    minFoodDistance = min(foodDistances) if foodDistances else 0

    # Calculate distance to the closest ghost
    minGhostDistance = min(manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates)

    evaluation = currentGameState.getScore() - minFoodDistance + 0.1 * minGhostDistance

        # Adjust the evaluation based on the remaining scared time of ghosts
    for scaredTime in scaredTimes:
        if scaredTime is not None and scaredTime > 0:
            evaluation += 15  # Adjust the weight based on your preferences
    return evaluation


# Abbreviation
better = betterEvaluationFunction
