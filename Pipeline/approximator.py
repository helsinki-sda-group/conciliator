import numpy as np
import random
from deep_sea_treasure import DeepSeaTreasureV0
from deep_sea_treasure import FuelWrapper
import numpy as np
from weighting import normal_distance_weighting

class Approximator():   

    """
    Approximates actions' consequences and consquently rewards using Monte Carlo sampling and assumptions of deterministic transitions
    """

    def acceleration(self,action):
        """A function to return the acceleration of an action

        Args: 
            self (object): an approximator object
            action (int): the taken action of the approximator from a flat array

        Returns:
            a_x ([int]): the velocity in x-direction
            a_y ([int]): the velocity in y-direction
        """ 
        a_x, a_y = np.zeros(7), np.zeros(7)
        a_x[action % 7] = 1; a_y[(action - action % 7)// 7] = 1
        return a_x, a_y
    
    def collides(self, dst, next_pos):
        x, y = dst.sub_pos[0][0], dst.sub_pos[1][0]
        x_next, y_next = next_pos[0][0], next_pos[1][0]
		# If moving over the left wall
        if x_next < 0:
            return True
		# If moving over the upper edge
        if y_next < 0:
            return True
		# If moving over the right edge
        if x_next > len(dst.seabed) - 1:
            return True
		# If the submarine is not moving, no collision can occur
        if (x,y) == (x_next,y_next):
            return False
		# If moving left, order the coordinates
        if x_next < x:
            x, x_next = x_next,x
        indices = np.arange(0,len(dst.seabed),1,dtype=int)
		# The submarine ends up moving inside the seabed
        if np.any((x_next <= indices) & (y_next > dst.seabed)):
            return True
		# The submarine moves vertically and ends up inside the seabed
        if x_next == x:
            return y_next > dst.seabed[int(x)]
		# The submarine moves and a line from the previous location to 
		# the new location intercepts the seabed
		# Line coefficient for a line ax+by+c=0 which goes through points (x,y) and (x_next,y_next)
        a,b,c = y_next-y,-(x_next-x),(x_next-x)*y-(y_next-y)*x
		# Consider only x-values between x and x_next
        indices = indices[(indices >= x) & (indices <= x_next)]
		# Directional distance to each seabed point inside the domain of the line,
		# coordinates with larger y-xoordinates than the line have negative distance
		# dists = np.abs(a*indices + b*dst.seabed[indices] + c) / np.sqrt(a**2+b**2)
        line = lambda x: -(a*x + c) / b
		# If moving in a legal area
        if not np.all(line(indices[:-1]) < dst.seabed[indices][:-1]):
            return True
        if np.any(line(indices) > dst.seabed[indices]):
            return True
		# Fail safe
		#else:
        return False
    
    def reward_prediction(self, dst, state, action, cumulative_reward, preference):
        """A function to calculate the reward estimate of an action

        Args: 
            self (object): an Approximator object
            state ([float]): the state vector of velocity and relative locations
            action (int): the taken action of the approximator from a flat array
            treasures ([int]): the treasure chests' values as a list
            cumulative_reward ([float]): the reward gained so far in the state

        Returns:
            time (float): the prediction of time used in the action
            fuel (float): the prediction of fuel used in the action
            treasure (float): the prediction of the treasure gained by the action
        """ 
        action = np.array([action % 7 - 3, (action - action % 7) // 7 - 3])
        treasures = np.array(list(dst.treasures.values()))
        pos = dst.sub_pos
        predicted_velo = state[:,0] + action
        # If overspeed
        if np.any(np.abs(predicted_velo) > dst.max_vel):
            return (-np.inf, -np.inf, -np.inf)
        predicted_pos = pos + predicted_velo.reshape((2,1))
        predicted_locs = state[:,1:] - predicted_velo.reshape((2,1))
        preferred_treasures = treasures >= preference[0]
        weights = normal_distance_weighting([(0,0)],predicted_locs[:,preferred_treasures].T)[0,:]
        treasure = np.sum(weights*treasures[preferred_treasures])
        time = cumulative_reward[1] - 1
        fuel = cumulative_reward[2] - np.sum(action**2)
        # If collides
        if self.collides(dst,predicted_pos):
            return (-np.inf, -np.inf, -np.inf)
        # Otherwise regular estimate
        else:
            return (treasure, time, fuel)   

    def __init__(self, seed=123):
        """A function to initialize the Approximator

        Args: 
            self (object): an Approximator object
            seed (int): the random number generator's seed
        """ 
        random.seed(seed)
        self.mapping = lambda r: np.array([r[0]/20, np.exp(r[1]/10), np.exp(r[2]/10)])
        self.action_space_size = 49
    
    def next_action(self, preference, state, dst, cumulative_reward, priority):
        """A function to return the next best action based on the preference

        Args: 
            preference ([float]): the preferred reward vector
            state ([float]): the current state 
            dst (Object): the simulation environment object
            cumulative_reward ([float]): the collected rewards until the state

        Returns:
            action ([float]): the next action as action vector
        """ 
        
        utilities = []
        for i in range(self.action_space_size):
            treasure, time, fuel = self.reward_prediction(dst,state,i,cumulative_reward,preference)
            pred_reward = np.array([treasure,time,fuel])
            pred_utility = self.scalarisation(pred_reward, priority, preference)
            utilities.append(pred_utility)
        action_ind = np.argmax(utilities)
        pred_reward = self.reward_prediction(dst,state,action_ind,cumulative_reward,preference)
        pred_utility = self.scalarisation(pred_reward, priority, preference)
        action = self.acceleration(action_ind)
        return action

    def scalarisation(self, reward, weights, preference, p=1):
        """A function to calculate optimal policies based on the priority order as weights

        Args:
            self (self): an Approximator object
            r ([float]): the reward vector
            weights ([float]): the priority order
        """
        if np.any(np.isinf(reward)):
            utility = -np.inf
        else:
            #utility = 1/np.average(np.abs(self.mapping(preference) - self.mapping(reward))**p, weights=weights)**(1/p)
            utility = np.exp(-np.average(np.abs(self.mapping(preference) - self.mapping(reward))**p, weights=weights)**(1/p))
        return utility