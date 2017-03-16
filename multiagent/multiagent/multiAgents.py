# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import sys
from decimal import Decimal

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        '''
        Assigning different cost to each item on the layout
        Ghost is not desirable hence chose ghost as the one
        having the higher cost
        '''
        FOOD_COST = 10.0
        GHOST_COST = 20.0

        '''
        Get the scores of all the successors of this state
        '''
        utility = successorGameState.getScore()

        '''
        find the distance to the nearest ghost and calculate the total cost
        from the current distance
        '''
        distanceToGhost = util.manhattanDistance(newPos, newGhostStates[0].getPosition())
        foodlist =  newFood.asList()

        if distanceToGhost > 0:
            utility -= GHOST_COST / distanceToGhost

        distancesToFood = [manhattanDistance(newPos, x) for x in foodlist]
        if len(distancesToFood):
            utility += FOOD_COST / min(distancesToFood)

        return utility

def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        " *** YOUR CODE HERE *** "
        def search(state, depth, agent):

            totalagents = state.getNumAgents()

            '''
            check for the number of agents in the game and check
            whether the given depth is equal to depth required otherwise
            we go to the next depth with 0 agents
            '''
            if agent == totalagents:
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return search(state, depth + 1, 0)
            else:
                '''
                Returns a list of legal actions for an agent
                '''
                actions = state.getLegalActions(agent)
                '''
                if the list of actions are not empty return the evaluation
                function on those list of actions
                '''
                if not actions:
                    return self.evaluationFunction(state)
                '''
                Get all the list of next states and append the successors of
                each state to each next state
                '''
                next_states = []
                for action in actions:
                    next_states.append(search(state.generateSuccessor(agent, action), depth, agent + 1))

                '''
                If the number of agents are 0, then we return the max of all next states
                else we return the min of next states
                '''
                if agent == 0:
                    return max(next_states)
                else:
                    return min(next_states)

        actions = gameState.getLegalActions(0)
        return max(actions, key=lambda x: search(gameState.generateSuccessor(0, x), 1, 1))
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):

        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ''' INITIALIZATION OF CONSTANTS '''
        '''
        We represent the number of ghosts, totalScore, alpha and beta values here
        '''
        ghosts = gameState.getNumAgents()-1
        totalScore = -Decimal("inf")
        alpha = -Decimal("inf")
        beta = Decimal("inf")

        actions = gameState.getLegalActions(0)

        '''
        We calculate the alpha and beta scores of each action and then return it
        '''
        for action in actions:
            copiedvalue = totalScore
            totalScore = max(totalScore, self.MinValue(gameState.generateSuccessor(0, action), self.depth, ghosts,1,alpha,beta))

            if totalScore > copiedvalue:
                Selectedaction = action

            '''
            Here we calculate the max of alpha and totalScore to determon
            '''
            alpha = max(alpha,totalScore)
        '''
        If the totalscore is infinity then we return STOP command
        '''
        if totalScore == -Decimal("inf"):
            return Directions.STOP

        return Selectedaction

    def MaxValue(self,gameState,depth,ghosts,alpha,beta):

        '''
        :param gameState: input is the gamestate which is the initial config
        :param depth: depth is the max depth of the game
        :param ghosts: number of ghosts
        :param alpha: the maximizer variable
        :param beta: the minimizer variable
        :return: the final value of the function which maximizes the gamestate
        '''

        '''
        Initialization of the Game
        '''
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        '''
        We define games here to calculate the totalScore of the function
        '''
        totalScore = -Decimal("inf")
        '''
        Defines all set of legal actions
        '''
        actions = gameState.getLegalActions(0)

        for action in actions:
            totalScore = max(totalScore, self.MinValue(gameState.generateSuccessor(0, action), depth, ghosts,1,alpha,beta))
            if totalScore > beta:
                return totalScore
            alpha = max(alpha,totalScore)

        return totalScore

    def MinValue(self, gameState,depth,ghosts,agentIndex,alpha,beta):
        '''
        :param gameState: input is the gamestate which is the initial config
        :param depth: depth is the max depth of the game
        :param ghosts: number of ghosts
        :param alpha: the maximizer variable
        :param beta: the minimizer variable
        :return: the final value of the function which minimizes the gamestate
        '''
        '''
        Initialization of the Game
        '''
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        '''
        We define games here to calculate the totalScore of the function
        '''
        totalScore = Decimal("inf")
        '''
        Defines all set of legal actions
        '''
        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            if ghosts == agentIndex:
                totalScore = min(totalScore, self.MaxValue(gameState.generateSuccessor(agentIndex, action), depth-1, ghosts,alpha,beta))

            else:
                totalScore = min(totalScore, self.MinValue(gameState.generateSuccessor(agentIndex, action), depth, ghosts,agentIndex+1,alpha,beta))
            '''
            If the value returned is less than alpha then we return the value
            to minimize the alpha,
            otherwise we return the beta value of the function
            '''
            if (totalScore < alpha):
                return totalScore
            '''
            We store the minimum of beta and totalscore in beta variable
            '''
            beta = min(beta,totalScore)

        return totalScore
        #util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        return self.ExpectiMax(gameState, 1, 0)

    def ExpectiMax(self, gameState, currentDepth, agentIndex):
        '''
        :param gameState: current gamestate
        :param currentDepth: the current depth of the game
        :param agentIndex: number of the agent
        :return: Returns the expectimax action using self.depth and self.evaluationFunction
        '''
        "*** YOUR CODE HERE ***"
        "Terminal condition"
        if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        "expectimax algorithm"
        Moves = []
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            if action != 'Stop':
                Moves.append(action)

        # update next depth
        nextAgentIndex = agentIndex + 1
        nextDepth = currentDepth

        agents = gameState.getNumAgents()
        if nextAgentIndex >= agents:
            nextAgentIndex = 0
            nextDepth += 1

        '''
        We append the succesors to the results list
        '''
        results = []
        for action in Moves:
            results.append(self.ExpectiMax(gameState.generateSuccessor(agentIndex, action) , nextDepth, nextAgentIndex))

        indexes = []
        if agentIndex == 0 and currentDepth == 1:
            '''
            pacman first move is the maximum among all the results
            '''
            bestMove = max(results)
            for index in range(len(results)):
                if results[index] == bestMove:
                    indexes.append(index)

            # Pick randomly among the best
            chosenIndex = random.choice(indexes)
            #print 'pacman %d' % bestMove
            return Moves[chosenIndex]

        if agentIndex == 0:
            Move = max(results)
            #print bestMove
            return Move

        else:
            "In ghost node, return the average(expected) value of action"
            '''
            Calculate the average of results
            '''
            Move = sum(results)/len(results)
            #print bestMove, sum(results), len(results)
            return Move
            #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    '''
    Terminal Condition check
    '''
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return - float("inf")

    '''
    Initializtion of variables
    '''
    totalScore = scoreEvaluationFunction(currentGameState)
    newFood = currentGameState.getFood()
    foodList = newFood.asList()
    closestfood = float("inf")

    for food in foodList:
        dist = util.manhattanDistance(food, currentGameState.getPacmanPosition())
        if (dist < closestfood):
            closestfood = dist

    ghosts = currentGameState.getNumAgents() - 1
    counter = 1
    distToGhost = float("inf")
    while counter <= ghosts:
        nextdist = util.manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(counter))
        distToGhost = min(distToGhost, nextdist)
        counter += 1

    '''
    Get all the distances to ghost and food and calculate the min
    of each of those distances
    '''
    totalScore += max(distToGhost, 5) * 2
    totalScore -= closestfood * 2
    capsulelocations = currentGameState.getCapsules()
    totalScore -= 5 * len(foodList)
    totalScore -= 4 * len(capsulelocations)
    
    return totalScore
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
