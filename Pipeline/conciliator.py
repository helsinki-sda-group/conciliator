import numpy as np
from matplotlib.pyplot import Figure
import matplotlib.pyplot as plt
from scipy.optimize import shgo
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class Conciliator():    

    def __init__(self, objectives, R, eps=1e-16, priority=None, filename="Preference"):
        """A function to perform one iteration of Conciliator on a given machine epsilon and reward vector

        Args:
            R ([float]): a reward vector
            self (self, optional): a Conciliator object
            eps ([float], optional): machine epsilon. Defaults to 1e-16.

        Returns:
        priority ([float]): the priority order chosen by the user
        preference ([float]): the reward vector preferred by the user
        transfer ([float]): the required transfer to meet the prefence of the user
        """    
        # Set the global variables
        self.eps = eps
        self.R = R
        # self.neg_rewards = self.R < 0
        # self.mapping = lambda r: np.array([np.exp(r[i]/20) if r[i]<0 else r[i] / 60 for i in range(len(r))])
        self.mapping = lambda r: np.array([r[0]/60, np.exp(r[1]/10), np.exp(r[2]/10)])
        #self.mapping = lambda r: np.array([np.exp(r[i]/20) if self.neg_rewards[i] else r[i] / 60 for i in range(len(r))])
        self.inv_mapping = lambda r: np.array([r[0]*60, 10*np.log(r[1]), 10*np.log(r[2])])
        #self.inv_mapping = lambda r : np.array(20*np.log(r[i]) if self.neg_rewards[i] else [r[i]*60 for i in range(len(r))])
        self.scaled_R = self.mapping(self.R)
        self.objectives = objectives
        self.filename = filename
        self.preference = np.zeros(len(self.R))
        self.transfer = np.zeros(len(self.R))
        self.priority=priority
        self.run()
        
    def run(self):
        """A function to run Conciliator steering's visual GUI

        Args:
            self (self, optional): a Conciliator object
        """    
        # Set the figure
        self.root = tk.Tk()
        self.root.title("Conciliator steering")
        self.bars = []
        n = len(self.R)
        self.fig, self.axs = plt.subplots(1, n)
        self.fig.set_size_inches(10,5)
        self.width = 1/n
        for i in range(n):
            self.axs[i].set_ylim(-abs(4*self.R[i]), abs(4*self.R[i]))
            self.axs[i].set_xlim(-0.5, 0.5)
            self.axs[i].bar(0,height=self.R[i],width=self.width,color='tab:blue',label="Original reward")
            bar1 = self.axs[i].bar(0,height=0,bottom=min(0,self.R[i]),width=self.width,color='tab:orange',label="Reward removed")
            bar2 = self.axs[i].bar(0,height=0,bottom=min(0,self.R[i]),width=self.width,color='tab:green',label="Reward added")
            self.axs[i].axhline(y=0, color='black', linestyle='--')
            self.axs[i].set_xticks([],[])
            self.axs[i].set_xlabel(self.objectives[i])
            self.bars.append((bar1, bar2))
            if i == n-1:
                self.axs[i].legend()

        # Create the reset button
        button = ttk.Button(text='Reset')
        button.bind('<Button>', self.reset)
        button.grid(row=len(self.R)+2,column=1)

        # Create the equal button
        button = ttk.Button(text='Equal')
        button.bind('<Button>', self.equal)
        button.grid(row=len(self.R)+2,column=2)

         # Create the "Done" button
        button = ttk.Button(text='Done')
        button.bind('<Button>', self.handle_done)
        button.grid(row=len(self.R)+2,column=3)

        # Set the canvas
        self.canvas = FigureCanvasTkAgg(self.fig,master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=5)

        # Set the sliders
        self.sliders = []
        for i in range(len(self.R)):
            label = ttk.Label(text=f'Objective #{i+1}: {self.objectives[i]}')
            label.grid(row=i+1,column=0)
            slider = ttk.Scale(self.root,from_=self.eps, to=100, orient='horizontal', length=800)
            slider.set(50)
            slider.bind(f"<ButtonRelease-1>", self.updateSlider)
            slider.grid(row=i+1,column=2)
            self.sliders.append(slider)
        self.slider_values = np.zeros(len(self.sliders))

        # A dynamically assigned priority
        if self.priority is None:
            self.root.mainloop()
        # A pre-defined priority for the tests
        else:
            self.update_transfer(self.priority)
            self.update_fig()
            self.save_figure()
            self.root.quit()

    def update_res(self):
        """A function to update the results

        Args:
            self (self): a Conciliator object
        """  
        # Enforce the constraints
        temp = self.scaled_R + self.transfer
        if temp[1] < np.exp(-1.5):
            temp[0] -= np.exp(-1.5) - temp[1]
            temp[1] = np.exp(-1.5)
        if temp[2] < np.exp(-3.0):
            temp[0] -= np.exp(-3.0) - temp[2]
            temp[2] = np.exp(-3.0)
 
        for i in range(1,3):
            if temp[i] > 1.0:
                temp[0] += temp[i] - 1.0
                temp[i] = 1.0

        self.preference = self.inv_mapping(temp)

    def handle_done(self,event):
        """A function to quit and save the figure when done

        Args:
            self (self): a Conciliator object
            event (event): a button click that indicates that the user is done with Conciliator steering
        """        
        self.root.quit()
        self.save_figure()

    def save_figure(self):
        current_directory_path = os.getcwd()
        file_name = f'{self.filename}.png'
        file_path = os.path.join(current_directory_path, file_name)
        subfolder_path = os.path.join(current_directory_path, 'Results')
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        file_path = os.path.join(subfolder_path, file_name)
        plt.savefig(file_path)

    # Function to be optimized
    def score(self,t,priority):
        """A function to optimize over

        Args:
            self (self): a Conciliator object
            t ([float]): a transfer vector

        Returns:
            loss (float): the squared sum difference between the preferred reward and empirical mean reward
        """    
        # Normalize the priority order        
        loss = np.sum(np.square(self.scaled_R + t - priority * np.mean(self.scaled_R)))
        return loss

    def optimize_transfer(self,priority,bounds=None):
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
        fun = lambda t: self.score(t,priority)
        if bounds is None:
            bounds = np.column_stack((-self.scaled_R, self.scaled_R))
        t_opt = shgo(fun,bounds,constraints=constraints).x
        self.transfer = t_opt
    
    def updateSlider(self, event):
        """A function to update the figure after the sliders are used

        Args:
            self (self): a Conciliator object
            event (event, optional): an event updating sliders in the GUI
        """    
        self.get_sliders()
        self.remove_bars()
        self.update_transfer(self.slider_values)
        self.update_fig()

    def get_sliders(self, set_eq=False):
        """A function to set new values to sliders
        Args:
            self (self): a Conciliator object
            set_eq ([float], optional): whether to set the sliders to equal
        """    
        for i in range(len(self.sliders)):
            if set_eq:
                self.sliders[i].set(50)
            self.slider_values[i] = self.sliders[i].get()

    def remove_bars(self):
        """A function to remove the bars from the figure when performing a reset

        Args:
            self (self): a Conciliator object
        """  
        for (bar1,bar2) in self.bars:
            bar1.remove()
            bar2.remove()

    def update_fig(self):
        """A function to update the figure

        Args:
            self (self): a Conciliator object
        """    
        self.fig.canvas.draw()
        self.root.update()

    def update_transfer(self, priority):
        """A function to update the transfer vector

        Args:
            self (self): a Conciliator object
        """
        priority = len(priority)*priority/np.sum(priority) if np.sum(priority) != 0 else np.ones(len(priority))        
        self.optimize_transfer(priority)
        self.update_res()
        for i in range(len(self.axs)):
            r = self.R[i]
            p = self.preference[i]
            
            if r >= 0:
                if p >= r:
                    remove_lims = (r,0) 
                    add_lims = (r,p-r) 
                else:
                    remove_lims = (p,r-p)
                    add_lims = (p,0)
            else:
                if p >= r:
                    remove_lims = (r,0) 
                    add_lims = (r,p-r) 
                else:
                    remove_lims = (p,r-p) 
                    add_lims = (p,0)
            bar1 = self.axs[i].bar(0,height=remove_lims[1],bottom=remove_lims[0],width=self.width,color='tab:orange',label="Reward removed")
            bar2 = self.axs[i].bar(0,height=add_lims[1],bottom=add_lims[0],width=self.width,color='tab:green',label="Reward added") 
            self.bars[i] = (bar1, bar2)

    def reset(self, event):
        """A function to reset the Conciliator to the starting state

        Args:
            self (self): a Conciliator object
        """  
        self.get_sliders(set_eq=True)
        self.remove_bars()
        self.priority = np.zeros(len(self.R))
        self.preference = np.zeros(len(self.R))
        self.transfer = np.zeros(len(self.R))
        for i in range(len(self.R)):
            bar1 = self.axs[i].bar(0,height=0,bottom=self.R[i],width=self.width,color='tab:orange',label="Reward removed")
            bar2 = self.axs[i].bar(0,height=0,bottom=self.R[i],width=self.width,color='tab:green',label="Reward added")
            self.axs[i].bar(0,height=self.R[i],width=self.width,color='tab:blue',label="Original reward")
            self.bars[i] = (bar1, bar2)
        self.update_fig() 

    def equal(self, event):
        """A function to set the priority order to equal and calculate the transfer

        Args:
            self (self): a Conciliator object
        """ 
        self.get_sliders(set_eq=True)
        self.priority = np.ones(len(self.R)) / len(self.R)
        self.remove_bars()
        self.update_transfer(self.slider_values)
        self.update_fig()