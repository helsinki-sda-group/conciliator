import numpy as np
from matplotlib.pyplot import Figure
from scipy.optimize import shgo
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import deep_sea_treasure
from deep_sea_treasure import DeepSeaTreasureV0
import pygame
import pandas as pd
from deep_sea_treasure import FuelWrapper

class Approximator(): 

    def init_dst():
    # Make sure experiment are reproducible, so people can use the exact same versions
        print(f"Using DST {deep_sea_treasure.__version__.VERSION} ({deep_sea_treasure.__version__.COMMIT_HASH})")

        dst: DeepSeaTreasureV0 =  FuelWrapper.new(DeepSeaTreasureV0.new(
            max_steps=1000,
            render_treasure_values=True,
        ))
        return dst  

    def error(self):
        error = np.sum(self.ground_truth - self.predictions)**2
        col_labels = [f"x{i}" for i in range(len(self.ground_truths))].insert("Velocity")
        df = pd.dataframe(zip(self.ground_truths, self.predictions), labels = col_labels)
        return df, error

    def update(self, action, curr_state, next_state):
        delta = next_state - curr_state
        self.predictions[(action, delta)] += 1
        pass

    def acceleration(self,action):
        a_x, a_y = np.zeros(7), np.zeros(7)
        a_x[action[0]] = 1; a_y[action[1]] = 1
        return a_x, a_y

    def rewards(self,action):
        distances = np.linalg.norm(self.predictions[:,1:11], axis=0)
        mask = distances == 0
        if np.any(mask):
            chest_index = np.where(mask)[0]
            chest_reward = chest_values[chest_index]
        else:
            chest_reward=0
        time = -1
        fuel = -1

    def ground_truth(self,state):
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
        if len(self.unused_actions) == 0:
            self.unused_actions = np.range(1,49)
        return random.sample(self.unused_actions)          

    # TODO: code the next action's choice for Conciliator
    def match():
        pass

    def state(self):
        return (self.dst.sub_pos, self.dst.sub_vel)

    def __init__(self):
        unused_actions = np.range(1,49)
        predictions = np.array()
        ground_truths = self.ground_truth()
        dst = self.init_dst()
        app = self.train(self)
        error, df = self.error(app)
        return app, error, df
    

        # TODO: code the approximators and their training/testing
    def train(self, n_iters, thresh):
        stop: bool = False
        iters: int = 0
        time_reward: int = 0
        while not stop:
            action = self.explore()
            current_state = self.state(self)
            _, reward, done, debug_info = self.dst.step(action)
            time_reward += int(reward[1])
            next_state = self.state(self)
            self.update(self, action, current_state, next_state)
            
            if done:
                time_reward=0
                error, df = self.error(self)
                if iters >= n_iters or error <= thresh:
                    print(df.info)
                    print(df)
                    print(f"Iterations done: {n_iters}")
                    stop = True

            if not stop:
                iters += 1
            
            if done:
                self.dst.reset()
        pass