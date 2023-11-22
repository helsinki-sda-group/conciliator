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


    def error(self):
        """A function to return the error of the approximator

        Args: 
            self (object): an approximator object

        Returns:
            df (DataFrame): a dataframe of predicted velocities and ground truth velocities
            error (float): a MSE bteween predicted and ground truth velocities
        """ 
        error = np.sum(self.ground_truth - self.predictions)**2
        col_labels = [f"x{i}" for i in range(len(self.ground_truths))].insert("Velocity")
        df = pd.dataframe(zip(self.ground_truths, self.predictions), labels = col_labels)
        return df, error

    def update(self, action, curr_state, next_state):
        """A function to return the error of the approximator

        Args: 
            self (object): an approximator object
            action (Array[Int]): the taken action of the approximator
            curr_state ([Float]): the current state
            next_state ([Float]): the next state after the action has been taken
        """ 
        delta = next_state[:,0] - curr_state[:,0]
        self.predictions[(action, delta)] += 1
        pass

    def acceleration(self,action):
        """A function to return the acceleration of an action

        Args: 
            self (object): an approximator object
            action (Array[Int]): the taken action of the approximator

        Returns:
            a_x (Array[Int]): the velocity in x-direction
            a_y (Array[Int]): the velocity in y-direction
        """ 
        a_x, a_y = np.zeros(7), np.zeros(7)
        a_x[action[0]] = 1; a_y[action[1]] = 1
        return a_x, a_y

    def ground_truth(self,state):
        """A function to construct the ground truth of the approximator

        Args: 
            self (object): an approximator object
            action (Array[Int]): the taken action of the approximator

        Returns:
            a_x (Array[Int]): the velocity in x-direction
            a_y (Array[Int]): the velocity in y-direction
        """ 
        velo_x, velo_y = state[0][0], state[0][1]
        gts = []
        for i in range(0,6):
            for j in range(0,6):
                gt = np.zeros_like(state)
                a_x, a_y = self.acceleration((i,j))
                gt[0,0] = velo_x + a_x
                gt[0,1] = velo_y + a_y
                for c in range(1,11):
                    gt[c,0] = gt[c,0] + gt[0,0]
                    gt[c,1] = gt[c,1] + gt[0,1]
                    gts.append(gt)
        return np.array(gt)
    
    def explore(self):
        """A function to return the error of the approximator

        Args: 
            self (object): an approximator object

        Returns:
            action (tuple): a random acceleration in x- and y-directions
        """ 
        if len(self.unused_actions) == 0:
            self.unused_actions = np.range(1,49)
        action = random.sample(self.unused_actions)
        return action          

    def reward_prediction(self, state, action):
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
        predicted_state = state + self.predictions[action]
        distances = np.linalg.norm(predicted_state, axis=0)
        mask = distances == 0
        if np.any(mask):
            chest_index = np.where(mask)[0]
            chest_rewards = zip(*self.dst.treasure_values()[1])
            treasure = chest_rewards[chest_index]
        else:
            treasure = 0
        time = -1
        fuel = -0.1
        return time, fuel, treasure

    def state(self, dst):
        """A function to return the error of the approximator

        Args: 
            self (object): an approximator object
            action (Array[Int]): the taken action of the approximator

        Returns:
            a_x (Array[Int]): the velocity in x-direction
            a_y (Array[Int]): the velocity in y-direction
        """ 
        return (self.dst.sub_pos, self.dst.sub_vel)

    def __init__(self, train_iters = 100, threshold=100):
        """A function to return the error of the approximator

        Args: 
            n_iters (int): the number of trajectories the approximator conducts while training
            thresh (float): the error threshold for MSE

        Returns:
            predictions (dict(tuple,int)): the predictions of actions' consequences as action-change -pairs
            error (float): the MSE error of predictions
            df (Pandas DataFrame): the predictions and ground truths
        """ 
        unused_actions = np.range(1,49)
        predictions = np.array()
        self.train(self, n_iters=train_iters, thresh=threshold)
        error, df = self.error(self)
        return predictions, error, df
    
    def train(self, n_iters = 100, thresh = 100):
        """A function to train the approximator

        Args: 
            n_iters (int): the number of trajectories the approximator conducts while training
            thresh (float): the error threshold for MSE
        """ 
        # Make sure experiment are reproducible, so people can use the exact same versions
        print(f"Using DST {deep_sea_treasure.__version__.VERSION} ({deep_sea_treasure.__version__.COMMIT_HASH})")

        dst: DeepSeaTreasureV0 =  FuelWrapper.new(DeepSeaTreasureV0.new(
            max_steps=1000,
            render_treasure_values=True,
        ))
        stop: bool = False
        iters: int = 0
        time_reward: int = 0
        while not stop:
            action = self.explore(self)
            current_state = self.state(self,dst)
            _, reward, done, debug_info = dst.step(action)
            time_reward += int(reward[1])
            next_state = self.state(self, dst)
            self.update(self, dst, action, current_state, next_state)
            
            if done:
                time_reward=0
                error, df = self.error(self)
                print(f"Iterations done: {iters}")
                if iters >= n_iters or error <= thresh:
                    print(df.info)
                    print(df)
                    stop = True

            if not stop:
                iters += 1
            
            if done:
                self.dst.reset()
        pass