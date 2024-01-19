import numpy as np
import random
from deep_sea_treasure import DeepSeaTreasureV0
from deep_sea_treasure import FuelWrapper
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

def shepard_weights(points,coords,p=1,metric='cityblock'):
    dists = cdist(points,coords,metric=metric)
    zero_rows, zero_columns = np.where(dists == 0.0)
    if len(zero_rows) != 0:
        dists = np.delete(dists,zero_rows,axis=0)             
    log_dists = p * np.log(dists)
    log_products = np.sum(log_dists,axis=1).reshape((-1,1)) - log_dists
    log_weights = log_products - logsumexp(log_products,axis=1).reshape((-1,1))
    weights = np.exp(log_weights)
    for i in range(len(zero_rows)):
        temp = np.zeros(len(coords))
        temp[zero_columns[i]] = 1.0
        weights = np.insert(weights,zero_rows[i],temp,axis=0)
    return weights

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

    def reward_prediction(self, dst, state, action, cumulative_reward):
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
        max_vel = np.sqrt(np.sum(np.square(dst.max_vel)))
        if np.any(np.abs(predicted_velo) > max_vel):
            return (-np.inf, -np.inf, -np.inf)
        predicted_velo[np.abs(predicted_velo) > max_vel] = max_vel * np.sign(predicted_velo[np.abs(predicted_velo) > max_vel])
        predicted_pos = pos + predicted_velo.reshape((2,1))
        predicted_locs = state[:,1:] - predicted_velo.reshape((2,1))
        weights = shepard_weights([(0,0)],predicted_locs.T)[0,:]
        treasure = np.average(treasures, weights=weights)
        # Regular time and fuel
        time = cumulative_reward[1] - 1
        fuel = cumulative_reward[2] - np.sum(action**2)
        x_next, y_next = float(predicted_pos[0]), -float(predicted_pos[1])
        # Check for seabed vertically and horizontally, and then the edges
        if np.any(((predicted_locs[0,:] >= 0) & (predicted_locs[1,:] < 0)) | (np.any((predicted_locs[0,:] > 0) & (predicted_locs[1,:] <= 0)))) or (x_next < 0) or (x_next >= 10) or (y_next > 0) or (y_next < -10):
            return (-np.inf, -np.inf, -np.inf)
        x, y =  float(dst.sub_pos[0]), -float(dst.sub_pos[1])
        treasure_coords = np.array(list(dst.treasures.keys()))
        # Check for the chests
        if np.any((x_next == treasure_coords[:,0]) & (-y_next == treasure_coords[:,1])):
            return (treasure, time, fuel)
        # Check if not stayed put
        elif (x,y) != (x_next,y_next):
            a,b,c = y_next-y,-(x_next-x),(x_next-x)*y-(y_next-y)*x
            dists = (-a*treasure_coords[:,0] - b*treasure_coords[:,1] + c) / np.sqrt(a**2+b**2)
            # Check if move intercepts seabed or chest
            if np.any((dists <= 0) & (dists > -np.sqrt(2)/2)) & np.any((predicted_locs[0,:] > 0) & (predicted_locs[1,:] < 0)):
                return (-np.inf, -np.inf, -np.inf)
        return (treasure, time, fuel)

    def __init__(self, seed=123):
        """A function to initialize the Approximator

        Args: 
            self (object): an Approximator object
            seed (int): the random number generator's seed
        """ 
        random.seed(seed)
        self.mapping = lambda r: np.array([r[0]/60, np.exp(r[1]/10), np.exp(r[2]/10)])
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
            treasure, time, fuel = self.reward_prediction(dst,state,i,cumulative_reward)
            pred_reward = np.array([treasure,time,fuel])
            pred_utility = self.scalarisation(pred_reward, priority, preference)
            utilities.append(pred_utility)
        action_ind = np.argmax(utilities)
        pred_reward = self.reward_prediction(dst,state,action_ind,cumulative_reward)
        pred_utility = self.scalarisation(pred_reward, priority, preference)
        action = self.acceleration(action_ind)
        return action

    def scalarisation(self, reward, weights, preference, p=1):
        """A function to calculate optimal policies based on the priority order as weights

        Args:
            self (self): an Approximator object
            r ((float)): the reward vector
            weights ((float)): the priority order
        """
        if np.any(np.isinf(reward)):
            utility = -np.inf
        else:
            utility = 1/np.average(np.abs(self.mapping(preference) - self.mapping(reward))**p, weights=weights)**(1/p)
        return utility