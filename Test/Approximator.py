import numpy as np
from matplotlib.pyplot import Figure
from scipy.optimize import shgo
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Approximator():    

    def __init__(self):
        pass

    # TODO: code the error for approximators
    def app_error():
        pass

    # TODO: code the prediction for approximators
    def app_predict():
        pass

    # TODO: code the next action's choice for approximator
    def next_app_action():
        pass

    # TODO: code the next action's choice for Conciliator
    def next_con_action():
        pass

        # TODO: code the approximators and their training/testing
    def run_approximators(n_iters, thresh):
        dst = init_dst()
        stop: bool = False
        iters: int = 0
        time_reward: int = 0
        while not stop:
            action = next_app_action()
            pred = app_predict(action)
            _, reward, done, debug_info = dst.step(action)
            time_reward += int(reward[1])
            
            if done:
                time_reward=0
                error = app_error(reward, pred)
                if iters >= n_iters or error <= thresh:
                    stop = True

            if not stop:
                iters += 1
            
            if done:
                dst.reset()
            
        pass