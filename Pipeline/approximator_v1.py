import numpy as np
import random
import deep_sea_treasure
from deep_sea_treasure import DeepSeaTreasureV0
import pandas as pd
from deep_sea_treasure import FuelWrapper

class Approximator():   

    """
    Approximates actions' consequences and consquently rewards using Monte Carlo sampling and assumptions of deterministic transitions
    """

    def update(self, action, curr_state, next_state):
        """A function to update the prediction of approximator

        Args: 
            self (object): an approximator object
            action (Array[Int]): the taken action of the approximator
            curr_state ([Float]): the current state
            next_state ([Float]): the next state after the action has been taken
        """ 
        delta = next_state[:,0] - curr_state[:,0]
        self.predictions[action] = delta

    def acceleration(self,action):
        """A function to return the acceleration of an action

        Args: 
            self (object): an approximator object
            action (Int): the taken action of the approximator from a flat array

        Returns:
            a_x (Array[Int]): the velocity in x-direction
            a_y (Array[Int]): the velocity in y-direction
        """ 
        a_x, a_y = np.zeros(7), np.zeros(7)
        print("Act", action, "Mod", action % 7, "Div", action // 7)
        a_x[action % 7] = 1; a_y[action // 7] = 1
        print(a_x, a_y)
        return a_x, a_y
    
    def explore(self):
        """A function to return the error of the approximator

        Args: 
            self (object): an approximator object

        Returns:
            action (tuple): a random acceleration in x- and y-directions
        """ 
        action_ind = random.sample(self.sampled_actions,1)[0]
        action = self.acceleration(action_ind)
        return action, action_ind          

    def reward_prediction(self, state, action, treasures, running_reward):
        """A function to calculate the reward estimate of an action

        Args: 
            self (object): an approximator object
            state (Array[float]): the state vector of velocity and relative locations
            action (Array[Int]): the taken action of the approximator

        Returns:
            time (float): the prediction of time used in the action
            fuel (float): the prediction of fuel used in the action
            treasure (float): the prediction of the treasure gained by the action
        """ 
        predicted_velo = state[:,0] + self.predictions[action]
        predicted_loc = state[:,1:] + np.reshape(predicted_velo,(2,1))
        distances = np.linalg.norm(predicted_loc, axis=0)
        treasure = np.median(np.mean(treasures))
                             #1/distances/np.sum(distances))
        time = running_reward[1]-1
        fuel = running_reward[2]+np.sum((self.predictions[action]**2))
        return treasure, time, fuel

    def state(self, dst):
        """A function to return the error of the approximator

        Args: 
            self (object): an approximator object
            action (Array[Int]): the taken action of the approximator

        Returns:
            sub_pos (Array[Int]): the velocity in x-direction
            sub_vel (Array[Int]): the velocity in y-direction
        """ 
        return (dst.sub_pos, dst.sub_vel)

    def __init__(self, n_iters = 1000):
        """A function to return the error of the approximator

        Args: 
            train_iters (int): the number of trajectories the approximator conducts while training
            thresh (float): the error threshold for MSE
        """ 
        print("Starting training!")
        self.predictions = {}
        self.mean = np.zeros(3)
        self.std = np.ones(3)
        random.seed(123)
        self.sampled_actions = random.choices(range(0,49),k=n_iters)
        self.train(n_iters=n_iters)
        print("Ready!")
    
    def next_action(self, preference, state, dst, running_reward):
        """A function to return the next best action based on the preference

        Args: 
            preference (Array[float]): the preferred reward vector
            state (Array[float]): the current state 

        Returns:
            next_action (Array[float]): the next action as action vector
        """ 
        
        next_action = (0,0)
        min_diff = np.inf
        for i in range(0,49):
            a_x, a_y = self.acceleration(i)
            pred_reward = np.array(self.reward_prediction(action=i,state=state, treasures=list(dst.treasures.values()),running_reward=running_reward))
            diff = np.sum(np.square(preference-pred_reward))
            if diff <= min_diff:
                next_action = (a_x, a_y)
                print(next_action)
                print(diff)
                min_diff = diff
            else:
                next_action = self.acceleration(random.choices(range(0,49),k=1)[0])
        return next_action
    
    def train(self, n_iters = 1000):
        """A function to train the approximator

        Args: 
            n_iters (int): the number of trajectories the approximator conducts while training
            thresh (float): the error threshold for MSE
        """ 

        # Make sure experiment are reproducible, so people can use the exact same versions
        print(f"Using DST {deep_sea_treasure.__version__.VERSION} ({deep_sea_treasure.__version__.COMMIT_HASH})")

        dst: DeepSeaTreasureV0 =  FuelWrapper.new(DeepSeaTreasureV0.new(
            max_steps=100,
            render_treasure_values=True,
        ))
        stop: bool = False
        iters: int = 0
        time_reward: int = 0
        rewards = []
        current_state = None
        while not stop:
            action, ind = self.explore()
            if current_state is None:
                current_state = np.zeros((2,11))
                current_state[:,1:] = np.column_stack(tuple(dst.treasures.keys()))
            next_state, reward, done, debug_info = dst.step(action)
            time_reward += int(reward[1])
            self.update(ind, current_state, next_state)
            current_state = next_state
            
            if done:
                iters += 1
                received_reward = [reward[0], time_reward+100, reward[2]+30]
                rewards.append(received_reward)
                time_reward=0

            if not stop:
                iters += 1
            
            if done:
                iters += 1
                dst.reset()
                if iters >= n_iters:
                    stop = True

        self.mean = np.mean(rewards,axis=0)
        self.std = np.std(rewards,axis=0)

    # TODO: policy calculation
    def policy_fit(self, baseline_rew, mean, std, cond="SER", n_pol=10):
        """A function to calculate optimal policies based on the scalarisation weights

        Args:
            self (self): a Conciliator object
            cond (String): the optimaility criterion
        """
        normed = (baseline_rew - mean) / std
        weights = self.priority_history
        if self.priority_history.ndim == 2:
            weights = np.mean(self.priority_history, axis=0)
        scalarisation = np.sum(weights*normed)
        return scalarisation