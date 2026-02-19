"""
Let me start with a markov chain
"""
import random
from matplotlib import pyplot as plt

class MarkovChain:
    def __init__(self, states, transitionMatrix):
        self.states = states 
        self.transitionMatrix = transitionMatrix
        self.event = []
        self.stateIter = {}


    def transition(self, currentState):
        nextState = random.choices(self.states, self.transitionMatrix[currentState], k = 1)
        self.event.append(nextState[0])
        self.stateIter[nextState[0]] += 1
        return nextState[0]

    def model(self, initialState, itr):
        for i in self.states:
            self.stateIter[i] = 0
            self.stateIter[initialState] += 1
        state = int(initialState)
        self.event.append(initialState)
        for i in range(0, itr):
            state = self.transition(state);
            



m1 = MarkovChain([0,1,2], [[0.1,0.2,0.7],[0, 0.6, 0.4], [0.1, 0.3, 0.6]])
m1.model(0, 1000)
xaxis = []
yaxis = []
for i in range(3):
    xaxis.append(i)
    yaxis.append(m1.stateIter[i])

plt.bar(xaxis, yaxis)
plt.show()
print(m1.stateIter)
