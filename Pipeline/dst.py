import numpy as np
import time
import sys
import os
#import deep_sea_treasure
#from deep_sea_treasure import DeepSeaTreasureV0
#from deep_sea_treasure import FuelWrapper
filepath = os.path.join(os.getcwd(), "deep_sea_treasure_v2")
if filepath not in sys.path:
    sys.path.append(filepath)
import deep_sea_treasure_v2
from deep_sea_treasure_v2 import DeepSeaTreasureV0, FuelWrapper
import pygame
import pandas as pd
import conciliator as Con 
import approximator as App
import json
from codecarbon import OfflineEmissionsTracker

def read_paretos(file):
    f = open(file)
    data = json.load(f)['pareto_front']
    pareto_front = {}
    j = 0
    rew_matrix = pd.DataFrame(columns=["Treasure", "Time", "Fuel"])
    for i in data:
        rew_matrix.loc[j] = [i['treasure'], i['time'], i['fuel']]
        pareto_front[j] = i['action_sequence']
        j+=1
    f.close()
    return pareto_front, rew_matrix
    
def init_dst():
    # Make sure experiment are reproducible, so people can use the exact same versions
    print(f"Using DST {deep_sea_treasure_v2.__version__.VERSION} ({deep_sea_treasure_v2.__version__.COMMIT_HASH})")

    dst: DeepSeaTreasureV0 =  FuelWrapper.new(DeepSeaTreasureV0.new(
        max_steps=50,
        render_treasure_values=True,
        max_velocity=4.0,
        implicit_collision_constraint=False,
        render_grid=True
    ))
    return dst    

def print_results(received, preferred, actions, pareto_front, pareto_rew_matrix):
    print("Actions:", actions)
    labels = ["treasure", "time", "fuel"]
    rews = {'received': received, 'preferred': preferred, 'difference': received-preferred}
    rew_df = pd.DataFrame(data=rews, index=labels)
    pd.set_option("display.precision", 3)
    print("\nDifference between the preferred and received rewards and Pareto optimal policies:")
    print(rew_df)
    rew_diffs = np.subtract(pareto_rew_matrix, np.reshape(received, (1,3)))
    pareto_sim = pd.DataFrame(data={labels[0]: rew_diffs.iloc[:,0], labels[1]: rew_diffs.iloc[:,1], labels[2]: rew_diffs.iloc[:,2]})
    n = len(actions)
    metric = 0
    metrics = []
    for i in range(len(pareto_front)):
        seq = pareto_front[i]
        for j in range(min(len(seq), n)):
            if seq[j] == actions[j]: 
                metric +=1
        metrics.append(metric/n)
        metric = 0
    pareto_sim["Pareto policy ratio"] = metrics
    print("\nDifference to Pareto optimal solutions")
    print(pareto_sim)

def array_to_json(arr):
    return np.where(arr==1)[0][0] - 3

def main():
    i = 2
    if len(sys.argv) > 1:
        i = int(sys.argv[1])

    # Emissions
    tracker = OfflineEmissionsTracker(country_iso_code="FIN", output_dir = "Pipeline/Results/", output_file = f"emissions_{i}.csv", tracking_mode = "process")
    tracker.start()

    # Pareto front and baseline
    pareto_front, pareto_rew_matrix = read_paretos("Pipeline/data/3-objective.json")
    incomplete_pareto_rew_matrix = pareto_rew_matrix.drop(pareto_rew_matrix.tail(1).index,inplace = False)
    baseline = incomplete_pareto_rew_matrix.mean(axis=0).values

    # Testing
    # Priority profiles
    # 1. "eco": 1/10 treasure, 2/10 time and 7/10 fuel
    # 2. "gold digger": 98/100 treasure, 1/100 time and 1/100 fuel
    # 3. "balanced": 1/5 treasure, 2/5 time and 2/5 fuel
    priorities = [[1/10,2/10,7/10],[98/100,1/100,1/100],[1/5,2/5,2/5]]
    dst = init_dst()

    policy = []

    stop: bool = False
    time_reward: int = 0
    fuel_reward: int = 0
    current_state = np.concatenate((dst.sub_vel, np.array(list(dst.treasures.keys())).T), axis=1) 
    running_reward = np.array([0.,0.,0.])
    app = App.Approximator()
    human=False

    print(f"\nHello! Conciliator steering has started.\n")
    # Seek out a policy for each user profile 
    priority = np.array(priorities[i])
    print(f"Profile {i}: {priority}")
    con = Con.Conciliator(objectives=['Treasure','Time','Fuel'], R = baseline, filename=f"{i}", priority=priority)
    print(f"Preference: {con.preference}\n")
    print(f"Baseline: {baseline}")

    dst.render()
    time.sleep(1)
    while not stop:
        if human:
            events = pygame.event.get()
            action = (np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0]))
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = (np.asarray([0, 0, 1, 0, 0, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0]))
                    elif event.key == pygame.K_RIGHT:
                        action = (np.asarray([0, 0, 0, 0, 1, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0]))
                    if event.key == pygame.K_UP:
                        action = (np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0]))
                    elif event.key == pygame.K_DOWN:
                        action = (np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 0, 0, 1, 0, 0]))

                    if event.key in {pygame.K_ESCAPE}:
                        stop = True

                if event.type == pygame.QUIT:
                    stop = True
        else:
            action = app.next_action(con.preference,current_state,dst,running_reward,con.priority)

        json_action_x = array_to_json(action[0])
        json_action_y = array_to_json(action[1])
        policy.append([json_action_x,json_action_y])
        previous_velo = dst.sub_vel.flatten()
        next_state, reward, done, debug_info = dst.step(action)
        
        next_velo = dst.sub_vel.flatten()
        current_state = next_state
        time_reward += reward[1]
        fuel_reward += reward[2]
        running_reward += np.array([reward[0],reward[1],reward[2]])
        
        if done:
            received_rews = np.asarray([reward[0], time_reward, fuel_reward])
            print_results(received_rews, con.preference, policy, pareto_front, pareto_rew_matrix)
            time_reward = 0
            fuel_reward = 0
            policy = []
        
        if not stop:
            dst.render()
            time.sleep(1)
        
        if done:
            dst.reset()
            stop = True
        
    print(f"\nConciliator steering has ended. Bye!\n\n")

    # Emissions
    emissions = tracker.stop()*1000
    print(f"Emissions: {emissions} kg of CO2")


if __name__ == "__main__":
    main()