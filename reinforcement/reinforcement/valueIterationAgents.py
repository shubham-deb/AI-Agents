# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        '''
        run iters for a given number of iterations using the supplied
        discount factor.
        '''
        # currently the number of iterations is 0
        cur_iterations = 0;

        #perform value iteration
        while cur_iterations < self.iterations:
            cur_iterations += 1
            allValues = {}
            poss_states = mdp.getStates()
            for state in poss_states:
            #if there are no actions don't iterate
                new_val = None
                actions_possible = mdp.getPossibleActions(state)
                #iterate over actions and get their qvalues
                for action in actions_possible:
                    if new_val is None:
                        new_val = self.computeQValueFromValues(state, action)
                    #get the best seen value for that state and action
                    new_val = max(new_val, self.computeQValueFromValues(state, action))

                allValues[state] = new_val

            #update the policy with the best values
            for key in allValues.keys():
                self.values[key] = allValues[key]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        #get the Transition function and nextStates
        statepairs = self.mdp.getTransitionStatesAndProbs(state, action)
        for t, p in statepairs:
            # print "trans", trans
            # print "prob" , prob
            # print "reward", self.mdp.getReward(state, action, trans)
            # print "discount", self.discount
            new_val = self.values[t]
            if new_val is None:
                new_val = 0
            q_value += p * (self.mdp.getReward(state, action, t) + self.discount*new_val)
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        #return None if terminal
        if self.mdp.isTerminal(state):
            return None

        #get the legal actions
        actions = self.mdp.getPossibleActions(state)
        val = None
        final_val = None

        #iterate over the legal actions and compute the qvalues
        for action in actions:
            temp_val = val
            val = self.computeQValueFromValues(state,action)
            if temp_val is None or temp_val < val:
                final_val = action
            else:
                val = temp_val

        # return the best value
        return final_val

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
