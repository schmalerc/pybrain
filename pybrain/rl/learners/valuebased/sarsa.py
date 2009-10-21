__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner


class SARSA(ValueBasedLearner):
    
    offPolicy = False
    batchMode = True
    
    def __init__(self, alpha=0.5, gamma=0.99):
        ValueBasedLearner.__init__(self)

        self.alpha = alpha
        self.gamma = gamma
    
        self.laststate = None
        self.lastaction = None

    def learn(self):
        """ learn on the current dataset, for a single step.
        
            in batchMode, the algorithm goes through all the samples in the
            history and performs an update on each of them. if batchMode is
            False, only the last data sample is considered. The user himself
            has to make sure to keep the dataset consistent with the agent's 
            history.
        """
        
        if self.batchMode:
            samples = self.dataset
        else:
            samples = [[self.dataset.getSample()]]

        for seq in samples:
            for state, action, reward in seq:
           
                state = int(state)
                action = int(action)
        
                # first learning call has no last state: skip
                if self.laststate == None:
                    self.lastaction = action
                    self.laststate = state
                    continue
        
                qvalue = self.module.getValue(self.laststate, self.lastaction)
                qnext = self.module.getValue(state, action)
                self.module.updateValue(self.laststate, self.lastaction, qvalue + self.alpha * (reward + self.gamma * qnext - qvalue))
        
                # move state to oldstate
                self.laststate = state
                self.lastaction = action
