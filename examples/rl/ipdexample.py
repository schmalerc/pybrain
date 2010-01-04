#!/usr/bin/python

from pybrain import *
#from pybrain.agents.continuousAgents import RandomAgent 
from pybrain.rl.environments import Environment
from pybrain.rl.environments.episodic import EpisodicTask
from pybrain.rl.experiments import EpisodicExperiment
from scipy import array, mean,ones
import random as rnd
from pybrain.rl.learners import *
from pybrain.structure import SigmoidLayer
from pybrain.rl.learners.valuebased.interface import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, QLambda, SARSA
from pybrain.rl.explorers import BoltzmannExplorer


class IPDOpponent():

    def __init__(self, strategy):
        self.strategy = strategy
        self.latest_strategy= None
        self.lastaction = None
        self.lastobs = None
        rnd.seed()
    
    def integrateObservation(self, obs):
        self.lastobs = obs

    def getAction(self):
                
        if self.latest_strategy == 'defect':
            self.lastaction = [0]
        
        elif self.latest_strategy == 'cooperate':
            self.lastaction = [1]

        elif self.latest_strategy == 'random':
            self.lastaction = [rnd.choice([0,1])]

        elif self.latest_strategy == 'tft':
            if self.lastobs[0] == 2:
                self.lastaction = [1]
            else:
                self.lastaction = [self.lastobs[0]]
        
        return self.lastaction

    def reset(self):
        if self.strategy == 'vary':
            self.latest_strategy = rnd.choice(['defect', 'cooperate', 'random', 'tft'])
        else:
            self.latest_strategy = self.strategy
            
        


class IPDEnvironment(Environment):

    # D = 0 # defect
    # C = 1 # cooperate
    # I = 2  # initialization (undef)

    def __init__(self, opponent):
        self.opponent = opponent
        
        D = 0 # defect
        C = 1 # cooperate
        I = 2  # initialization (undef)
        # Payoff-Matrix
        #        D       C
        # D: (1,1)  (5,0)
        # C: (0,5)  (3,3)

        self.payoffMatrix = {
                (D, D) : (1,1),
                (C, D) : (0,5),
                (D, C) : (5,0),
                (C, C) : (3,3)
        }        
        
        self.reset()

    def reset(self):
        D = 0 # defect
        C = 1 # cooperate
        I = 2  # initialization (undef)
        
        self.actions = [I]
        # [total opponent,
        #  last action opponent,
        #  total agent,
        #  last action agent,
        #  reward agent]
        self.sensors = [0.0, I, 0.0, I, 0.0]
        self.opponent.reset()

    def getSensors(self):
        return self.sensors

    def performAction(self, action):
        
        self.actions = action[:]

        self.opponent.lastaction = None
        self.opponent.lastobs = None        
        self.opponent.integrateObservation(self.getSensors()[3:4])
        self.oppAction = self.opponent.getAction()

        oppGain, agentGain = self.payoffMatrix[(self.oppAction[0], self.actions[0])]
        self.sensors[0] += oppGain
        self.sensors[1] = self.oppAction[0]
        self.sensors[2] += agentGain
        self.sensors[3] = self.actions[0]
        self.sensors[4] = agentGain
        

class IPDTask(EpisodicTask):

    def __init__(self, env, episodeLength):
        EpisodicTask.__init__(self, env)
        #self.inDim = 1 
        #self.outDim = 1
        self.counter = 0
        self.history = []
        self.total = []
        self.episodeLength = episodeLength

    def getObservation(self):
        
        return [self.env.getSensors()[1]]   

    def getReward(self):

        return self.env.getSensors()[4]
        

    def isFinished(self):
        
        self.counter += 1
        if self.counter <= self.episodeLength:
            self.total.append( (self.env.sensors[0], self.env.sensors[2]) )
            return False
        else:
            self.total.append( (self.env.sensors[0], self.env.sensors[2]) )
            self.history.append( self.total )
            self.total = []
            return True

    def reset(self):
        EpisodicTask.reset(self)
        self.counter = 0



if __name__ == '__main__':

    # for plotting
    plotting = True 

    if plotting:
        import pylab    
        pylab.ion()   

    # create the ActionValueTable
    table = ActionValueTable(3, 2)
    table.initialize(1)

    # create agent with controller and learner
    learner = QLambda()
    #learner.explorer = BoltzmannExplorer()

    agent = LearningAgent(table, learner)
    agent.learner.explorer.epsilon = 0.5
    agent.learner.explorer.decay = 0.999995
    

    # opponent strategy 'vary' changes randomly between 'defect', 'cooperate', 'random', 'tft' during training
    opponent = IPDOpponent('vary')
    env = IPDEnvironment(opponent)
    task = IPDTask(env, 100)
    experiment = EpisodicExperiment(task, agent)


    best=0.0
    rew=0.0
    plot = []
    for updates in range(1000):

        # testing step
        #agent.disableLearning()
        experiment.doEpisodes(1)

        # append mean reward to sr array
        ret = []
        for n in range(agent.history.getNumSequences()):
            state, action, reward = agent.history.getSequence(n)
            ret.append( sum(reward, 0).item() )
        rew=mean(ret)
        plot.append(task.history[-1][-1])
        #print "State:", state
        #print "Action:", action
        #print "Reward:", reward
        if rew>best: best=rew
        print "\n"
        print "Best: ", best, "Reward %f\n" % rew, "epsilon: ", agent.learner.explorer.epsilon
        print "last opponent strategy:", env.opponent.latest_strategy
        
        print "             Defect        Cooperate"
        print "Defect:\t ", table.getActionValues(0)
        print "Coop.:\t ", table.getActionValues(1)
        print "Init:\t ", table.getActionValues(2)        

        if plotting and updates > 1:
            pylab.figure(1)
            pylab.clf()
            pylab.title('Reward')
            a = pylab.plot(array(plot)[:,0])
            b = pylab.plot(array(plot)[:,1])
            pylab.legend([b,a], ['agent', 'opponent'])
            pylab.draw()  

        
        #agent.enableLearning()
        agent.reset()

        # training step
        
        experiment.doEpisodes(50)
        agent.learn()
        agent.reset()

