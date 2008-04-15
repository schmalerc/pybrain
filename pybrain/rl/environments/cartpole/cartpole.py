__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from matplotlib.mlab import rk4 
from math import sin, cos
import time
from scipy import random

from pybrain.rl.environments.graphical import GraphicalEnvironment


class CartPoleEnvironment(GraphicalEnvironment):
    """ This environment implements the cart pole balancing benchmark, as stated in:
        Riedmiller, Peters, Schaal: "Evaluation of Policy Gradient Methods and
        Variants on the Cart-Pole Benchmark". ADPRL 2007.
        It implements a set of differential equations, solved with a 4th order
        Runge-Kutta method.
    """       
    
    # some physical constants
    g = 9.81
    l = 0.5
    mp = 0.1
    mc = 1.0
    dt = 0.02    
    
    def __init__(self, polelength = None):
        GraphicalEnvironment.__init__(self)
        if polelength != None:
            self.l = polelength
        
        # initialize the environment (randomly)
        self.reset()
        self.action = 0.0
        self.delay = False

    def getSensors(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation. The sensor return 
            vector has 4 elements: theta, theta', s, s' (s being the distance from the
            origin).
        """
        return self.sensors
                            
    def performAction(self, action):
        """ stores the desired action for the next runge-kutta step.
        """
        self.action = action
        self.step()

    def step(self):
        self.sensors = rk4(self._derivs, self.sensors, [0, self.dt])
        self.sensors = self.sensors[-1]
        if self.hasRenderer():
            self.getRenderer().updateData(self.sensors)
            if self.delay: 
                time.sleep(0.05)    
                        
    def reset(self):
        """ re-initializes the environment, setting the cart back in a random position.
        """
        angle = random.uniform(-0.2, 0.2)
        pos = random.uniform(-0.5, 0.5)
        self.sensors = (angle, 0.0, pos, 0.0)

    def _derivs(self, x, t): 
        """ This function is needed for the Runge-Kutta integration approximation method. It calculates the 
            derivatives of the state variables given in x. for each variable in x, it returns the first order
            derivative at time t.
        """
        F = self.action
        (theta, theta_, s, s_) = x
        u = theta_
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        u_ = (self.g*sin_theta*(self.mc+self.mp) - (F + self.mp*self.l*theta**2*sin_theta) * cos_theta) / (4/3*self.l*(self.mc+self.mp) - self.mp*self.l*cos_theta**2)
        v = s_
        v_ = (F - self.mp*self.l * (u_*cos_theta - (s_**2 * sin_theta))) / (self.mc+self.mp)     
        return (u, u_, v, v_)   
    
    def getPoleAngles(self):
        """ auxiliary access to just the pole angle(s), to be used by BalanceTask """
        return [self.sensors[0]]
        
    def getCartPosition(self):
        """ auxiliary access to just the cart position, to be used by BalanceTask """
        return self.sensors[2]

    def getInDim(self):
        return 1
        
    def getOutDim(self):
        return 4
    


        