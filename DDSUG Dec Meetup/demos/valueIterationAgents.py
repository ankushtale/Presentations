# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here

        "*** YOUR CODE HERE ***"
        # print(self.mdp.getStates()[2])
        # print(self.mdp.getPossibleActions(self.mdp.getStates()[2]))
        # print(self.mdp.getTransitionStatesAndProbs(self.mdp.getStates()[2], 'west'))
        # mdp.getReward(state, action, nextState)
        # mdp.isTerminal(state)

        prev={}
        for s in self.mdp.getStates():
            prev[s] = 0

        for i in range(self.iterations):        
            for s in self.mdp.getStates():
                if self.mdp.isTerminal(s)==False:
                    val = []
                    for a in self.mdp.getPossibleActions(s):
                        v = 0
                        for t in self.mdp.getTransitionStatesAndProbs(s,a):
                            # t[0] = state and t[1]=transition
                            v += t[1]*(self.mdp.getReward(s,a,t[0]) + self.discount*prev[t[0]])
                        val.append(v)

                    self.values[s] = max(val)
    
            for s in self.mdp.getStates():
                prev[s] = self.values[s]
                    
                    

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
        q=0
        for t in self.mdp.getTransitionStatesAndProbs(state,action):
            # t[0] = state and t[1]=prob
            q += t[1]*(self.mdp.getReward(state,action,t[0]) + (self.discount*self.values[t[0]]))

        return q

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state)==False:
            val,act = max([(self.computeQValueFromValues(state,a),a) for a in self.mdp.getPossibleActions(state)])    

            return act
        else:
            return None

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        prev={}
        sl = []
        for s in self.mdp.getStates():
            sl.append(s)
            prev[s] = 0
        l = len(sl)
        for i in range(self.iterations):
            if i < len(sl):  
                if self.mdp.isTerminal(sl[i])==False:
                    val = []
                    for a in self.mdp.getPossibleActions(sl[i]):
                        v = 0
                        for t in self.mdp.getTransitionStatesAndProbs(sl[i],a):
                            # t[0] = state and t[1]=transition
                            v += t[1]*(self.mdp.getReward(sl[i],a,t[0]) + self.discount*prev[t[0]])
                        val.append(v)

                    self.values[sl[i]] = max(val)
                prev[sl[i]] = self.values[sl[i]]
            else:
                if self.mdp.isTerminal(sl[i%l])==False:
                    val = []
                    for a in self.mdp.getPossibleActions(sl[i%l]):
                        v = 0
                        for t in self.mdp.getTransitionStatesAndProbs(sl[i%l],a):
                            # t[0] = state and t[1]=transition
                            v += t[1]*(self.mdp.getReward(sl[i%l],a,t[0]) + self.discount*prev[t[0]])
                        val.append(v)

                    self.values[sl[i%l]] = max(val)
                prev[sl[i%l]] = self.values[sl[i%l]]


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        pred = dict()
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s)==False:
                l=[]
                for a in self.mdp.getPossibleActions(s):
                    for t in self.mdp.getTransitionStatesAndProbs(s,a):
                        # t[0] = state and t[1]=transition
                        if t[1]!=0:            
                            key = t[0]
                            if key not in pred:
                                pred[key] = []
                            pred[key].append(s)
                         
        pq = util.PriorityQueue()

        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s)==False:
                qmax = max(self.computeQValueFromValues(s,a) for a in self.mdp.getPossibleActions(s))
                diff = abs(self.values[s] - qmax)
                pq.update(s,-diff)

        for i in range(self.iterations):
            if pq.isEmpty()==True:
                return None
                # break
            else:
                s = pq.pop()
                if self.mdp.isTerminal(s)==False:
                    qmax = max(self.computeQValueFromValues(s,a) for a in self.mdp.getPossibleActions(s))
                    self.values[s] = qmax

                    for p in pred[s]:
                        if self.mdp.isTerminal(p)==False:
                            qmax_pred = max(self.computeQValueFromValues(p,a) for a in self.mdp.getPossibleActions(p))
                            diff = abs(self.values[p] - qmax_pred)
                            if diff > self.theta:
                                pq.update(p,-diff)





