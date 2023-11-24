import numpy as np
from matplotlib.pyplot import Figure
from scipy.optimize import shgo
from tkinter import Tk,ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def normalize_priority(priority):
    """A function to normalize the priority such that the sum of components is the number of objectives

    Args:
        priority ([float]): priority order

    Returns:
        normal ([float]): normalized priority order
    """    
    normal = len(priority) * priority / np.sum(priority)
    return normal

def run_conciliator(eps=1e-16):
    """A function that runs the Conciliator steering

    Args:
        eps (float, optional): machine epsilon. Defaults to 1e-16.
    """    
    # Theoretical baseline reward vector
    R = np.array([2,1,6,3,5,8,2,7])
    # Get user feedback through GUI
    priority = graphical_feedback(R)
    # Optimize using new priorities
    t = optimize_transfer(R,priority,eps=eps)
    # Update reward
    preference = R + t
    # Return results
    return preference, R, t, priority
    

# Function to be optimized
def score(t,R,priority):
    """A function to optimize over

    Args:
        t ([int]): a transfer vector
        R ([float]]): a reward vector
        priority ([float]): a priority order

    Returns:
        pref ([float]): the preferred reward vector
    """    
    pref = np.sum(np.square(R + t - priority * np.mean(R)))
    return pref

def optimize_transfer(R,priority,bounds=None,eps=1e-16):
    """A function to optimize the reward transfer for the given priority order

    Args:
        R ([float]): the reward vector
        priority ([float]): a priority order
        bounds ([float], optional): bounds for the transfer. Defaults to None.
        eps (float, optional): machine epsilon. Defaults to 1e-16.

    Returns:
     t_opt ([float]): the transfer vector required to meet the preferred reward
    """    
    # Constraints for the optimization, the amount of reward must stay constant
    constraints = ({'type':'ineq','fun': lambda t: np.sum(t) + eps},
                   {'type':'ineq','fun': lambda t: -np.sum(t) + eps})
    # Formulate the optimal scaled transfer vector
    fun = lambda t: score(t,R,priority)
    if bounds is None:
        bounds = np.column_stack((np.full(len(R),-np.inf),np.full(len(R),np.inf)))
    t_opt = shgo(fun,bounds,constraints=constraints).x
    return t_opt

def graphical_feedback(R,eps=1e-16):
    """A function to initiate the interactive window

    Args:
        R ([float]): a reward vector
        eps (float, optional): machine epsilon. Defaults to 1e-16.

    Returns:
        sliders ([]): the sliders for the priority order
    """    
    # Initiate the window
    window = Tk()
    window.resizable(True, True)
    window.title("Reward transfer")
    # Optimize using default priority
    priority = np.ones(len(R))
    t = optimize_transfer(R,priority,eps=eps)

    # Plot the figure
    pos = t >= 0
    neg = t < 0
    fig = Figure(figsize=(10,5))
    ax = fig.gca()
    ax.set_xlabel("Objective #")
    ax.set_ylabel("Reward value")
    ax.set_ylim(0,np.sum(np.abs(R-np.mean(R))))
    ax.bar(range(1,len(R)+1),height=R,label="Original reward")
    ax.axhline(y=np.mean(R),linestyle='--',color='black', label="Reward mean")
    ax.axhline(y=0,linestyle='-',color='black')
    bars = [ax.bar(np.where(neg)[0]+1,height=t[neg],bottom=R[neg],color='tab:orange',label="Reward removed"),
            ax.bar(np.where(pos)[0]+1,height=t[pos],bottom=R[pos],color='tab:green',label="Reward added")]
    ax.legend(fontsize=10)
    
    # Set the canvas and sliders
    canvas = FigureCanvasTkAgg(fig,master=window)
    canvas.get_tk_widget().grid(row=0, column=0, columnspan=5)
    sliders = []
    
    # Event handlers
    def handle_slider(event):
        """A function to update the solutions ater sliding the slidets

        Args:
            event (event): a change in sliders, indicating a change in priority order
        """
        # Normalize the priroity order        
        priority = normalize_priority(np.array([sliders[n].get() for n in range(len(sliders))]))
        # Optimize the trasnfer given the priority order
        t = optimize_transfer(R, priority)
        # Update the histogram
        pos = t >= 0
        neg = t < 0
        bars[0].remove()
        bars[1].remove()
        bars[0] = ax.bar(np.where(neg)[0]+1,height=t[neg],bottom=R[neg],color='tab:orange',label="Reward removed")
        bars[1] = ax.bar(np.where(pos)[0]+1,height=t[pos],bottom=R[pos],color='tab:green',label="Reward added")
        fig.canvas.draw()
        window.update()
    
    def handle_done(event):
        """A function to return the results in print and save the figure

        Args:
            event (event): a button click that indicates that the user is done with Conciliator steering
        """        
        # Save results
        priority[:] = [sliders[n].get() for n in range(len(sliders))]
        fig.savefig("Preferred_reward.png")
        # Quit GUI window
        window.quit()
    
    # Create the sliders
    for n in range(len(R)):
        label = ttk.Label(text=f'Objective #{n+1}')
        label.grid(row=n+1,column=0)
        slider = ttk.Scale(window,from_=1e-16, to=100, orient='horizontal',
                           length=800)
        slider.grid(row=n+1,column=1)
        slider.set(50)
        slider.bind("<ButtonRelease-1>", handle_slider)
        sliders.append(slider)
    
    # Create the "Done" button
    button = ttk.Button(text='Done')
    button.bind('<Button>',handle_done)
    button.grid(row=n+2,column=1)
    
    # Run interactively
    window.mainloop()
    # Return new priorities
    return priority

def main():
    # Run the Conciliator steering
    preference, R, t, priority = run_conciliator()
    # Print the results to the user
    print("Chosen priority order:", priority)
    print("Preferred reward vector:", preference)
    print("Transfer vector:", t)
    print("Original reward vector:", R)
    print("Conciliator steering has ended. Bye!")


if __name__ == "__main__":
    main()