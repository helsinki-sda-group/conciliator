import numpy as np
from matplotlib.pyplot import Figure
from scipy.optimize import shgo
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Conciliator():    

    def __init__(self, R, eps=1e-16, priority=None):
        """A function to perform one iteration of Conciliator on a given machine epsilon and reward vector

        Args:
            R ([float]): a reward vector
            self (self. optional): a Conciliator object
            eps ([float], optional): machine epsilon. Defaults to 1e-16.

        Returns:
        priority ([float]): the priority order chosen by the user
        preference ([float]): the reward vector preferred by the user
        """    
        # Set the global variables
        self.eps = eps
        self.R = R
        if priority is None:
            self.priority = np.zeros(len(self.R))
        else:
            self.priority=priority
        self.preference = np.zeros(len(self.R))
        self.transfer = np.zeros(len(self.R))

        # Set the figure
        self.root = tk.Tk()
        self.root.title("Priority order")
        self.fig = Figure(figsize=(10,5))
        self.ax = self.fig.gca()
        self.ax.set_xlabel("Objective #")
        self.ax.set_ylabel("Reward value")
        self.ax.set_ylim(0,np.sum(np.abs(self.R-np.mean(self.R))))
        self.ax.bar(range(1,len(self.R)+1),height=self.R,color='tab:blue',label="Original reward")
        self.ax.axhline(y=np.mean(self.R),linestyle='--',color='black', label="Reward mean")
        self.ax.axhline(y=0,linestyle='-',color='black')
        bar1 = self.ax.bar(np.ones(len(self.R)),height=0,bottom=self.R,color='tab:orange',label="Reward removed")
        bar2 = self.ax.bar(np.ones(len(self.R)),height=0,bottom=self.R,color='tab:green',label="Reward added")
        self.bars = [bar1, bar2]
        self.ax.legend(fontsize=10)

        # Create the reset button
        button = ttk.Button(text='Reset')
        button.bind('<Button>', self.reset)
        button.grid(row=len(self.R)+2,column=1)

        # Create the equal button
        button = ttk.Button(text='Equal')
        button.bind('<Button>', self.equal)
        button.grid(row=len(self.R)+2,column=2)

        # Set the canvas
        self.canvas = FigureCanvasTkAgg(self.fig,master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=5)

        # Set the sliders
        self.sliders = []
        for i in range(len(self.R)):
            label = ttk.Label(text=f'Objective #{i+1}')
            label.grid(row=i+1,column=0)
            slider = ttk.Scale(self.root,from_=self.eps, to=100, orient='horizontal', length=800)
            slider.set(50)
            slider.bind(f"<ButtonRelease-1>", self.updateSlider)
            slider.grid(row=i+1,column=2)
            self.sliders.append(slider)
        self.slider_values = np.zeros(len(self.sliders))
        self.root.mainloop() 
        self.preference = self.R + self.transfer
        self.priority = self.slider_values / np.sum(self.slider_values)

    # Function to be optimized
    def score(self,t):
        """A function to optimize over

        Args:
            self (self): a Conciliator object
            t ([int]): a transfer vector

        Returns:
            pref ([float]): the preferred reward vector
        """    
        # Normalize the priority order        
        priority = len(self.slider_values) * self.slider_values / np.sum(self.slider_values)
        pref = np.sum(np.square(self.R + t - priority * np.mean(self.R)))
        return pref

    def optimize_transfer(self,bounds=None):
        """A function to optimize the reward transfer for the given priority order

        Args:
            self (self): a Conciliator object
            bounds ([float], optional): bounds for the transfer. Defaults to None.

        Returns:
        t_opt ([float]): the transfer vector required to meet the preferred reward
        """    
        # Constraints for the optimization, the amount of reward must stay constant
        constraints = ({'type':'ineq','fun': lambda t: np.sum(t) + self.eps},
                    {'type':'ineq','fun': lambda t: -np.sum(t) + self.eps})
        # Formulate the optimal scaled transfer vector
        fun = lambda t: self.score(t)
        if bounds is None:
            bounds = np.column_stack((np.full(len(self.R),-np.inf),np.full(len(self.R),np.inf)))
        t_opt = shgo(fun,bounds,constraints=constraints).x
        self.transfer = t_opt
    
    def updateSlider(self, event):
        """A function to update the figure after the sliders are updated

        Args:
            self (self): a Conciliator object
            event (event, optional): an event updating sliders in the GUI
        """    
        # Collect the priority
        self.get_sliders()
        # Optimize the trasnfer given the priority order and update the histogram
        self.remove_bars()
        self.update_transfer()
        self.update_fig()

    def get_sliders(self, set_eq=False):
        for i in range(len(self.sliders)):
            if set_eq:
                self.sliders[i].set(50)
            self.slider_values[i] = self.sliders[i].get()

    def remove_bars(self):
        self.bars[0].remove()
        self.bars[1].remove()

    def update_fig(self):
        self.fig.canvas.draw()
        self.root.update()

    def update_transfer(self):
        self.optimize_transfer()
        pos = self.transfer >= 0
        neg = self.transfer < 0
        bar1 = self.ax.bar(np.where(neg)[0]+1,height=self.transfer[neg],bottom=self.R[neg],color='tab:orange',label="Reward removed")
        bar2 = self.ax.bar(np.where(pos)[0]+1,height=self.transfer[pos],bottom=self.R[pos],color='tab:green',label="Reward added")
        self.bars = [bar1, bar2]

    def reset(self, event):
        self.get_sliders(set_eq=True)
        self.remove_bars()
        self.priority = np.zeros(len(self.R))
        self.preference = np.zeros(len(self.R))
        self.transfer = np.zeros(len(self.R))
        bar1 = self.ax.bar(np.ones(len(self.R)),height=0,bottom=self.R,color='tab:orange',label="Reward removed")
        bar2 = self.ax.bar(np.ones(len(self.R)),height=0,bottom=self.R,color='tab:green',label="Reward added")
        self.ax.bar(range(1,len(self.R)+1),height=self.R,color='tab:blue',label="Original reward")
        self.bars = [bar1, bar2]
        self.update_fig() 

    def equal(self, event):
        self.get_sliders(set_eq=True)
        self.priority = np.ones(len(self.R)) / len(self.R)
        # Optimize the trasnfer given the priority order and update the histogram
        self.remove_bars()
        self.update_transfer()
        self.update_fig()