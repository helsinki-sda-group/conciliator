import numpy as np
import time
import deep_sea_treasure
from deep_sea_treasure import DeepSeaTreasureV0
import pygame
import pandas as pd
from deep_sea_treasure import FuelWrapper
import conciliator_v2 as Con 
import approximator_v1 as App
import json

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
    print(f"Using DST {deep_sea_treasure.__version__.VERSION} ({deep_sea_treasure.__version__.COMMIT_HASH})")

    dst: DeepSeaTreasureV0 =  FuelWrapper.new(DeepSeaTreasureV0.new(
        max_steps=1000,
        render_treasure_values=True,
    ))
    return dst

def run(con, app, human, preference, test_iters=1, pareto_front=None, pareto_rew_matrix=None):
    dst = init_dst()

    dst.render()
    actions = []
    iters = 0

    stop: bool = False
    time_reward: int = 0
    current_state = None

    while not stop:
        if current_state is None:
            current_state = np.zeros((2,11))
            current_state[:,1:] = np.column_stack(tuple(dst.treasures.keys()))
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
            action = app.next_action(preference,current_state,dst)

        print(action)
        json_action1 = np.where(action[0]==1)[0][0] - 3
        print(json_action1)
        json_action2 = np.where(action[1]==1)[0][0] - 3
        print(json_action2)
        actions.append([json_action1,json_action2])
        time.sleep(0.10)
        next_state, reward, done, debug_info = dst.step(action)
        current_state = next_state
        time_reward += int(reward[1])
        
        if done:
            received_rews = np.asarray([int(reward[0]), time_reward, int(reward[2])])
            print_results(received=received_rews, preferred=preference, actions=actions, objectives=con.objectives, pareto_front=pareto_front, pareto_rew_matrix=pareto_rew_matrix)
            time_reward = 0
        
        if not stop:
                dst.render()
                time.sleep(0.25)

        if done:
            iters += 1
            dst.reset()
            if iters >= test_iters:
                stop = True
    
    print(f"Conciliator steering has ended. Bye!\n\n")

def print_results(received, preferred, actions, objectives, pareto_front=None, pareto_rew_matrix=None):
    if pareto_front is None:
        pareto_front, pareto_rew_matrix = read_paretos("Pipeline/data/3-objective.json")
    rews = {'received': received, 'preferred': preferred, 'difference': received-preferred}
    rew_df = pd.DataFrame(data=rews, index=objectives)
    pd.set_option("display.precision", 3)
    print("\nDifference between the preferred and received reward ")
    print(rew_df)
    rew_diff = pareto_rew_matrix['Treasure'] == received[0]
    pareto_sim = pd.DataFrame(data={'Same chest': rew_diff})
    n = len(actions)
    metric = 0
    metrics = []
    for i in range(len(pareto_front)):
        seq = pareto_front[i]
        for j in range(min(len(seq), n)):
            if seq[j] == actions[j]: 
                metric +=1
        metrics.append(metric/n*100)
        metric = 0
    pareto_sim["% of same actions w.r.t. your own"] = metrics
    print("\n\nDifference to Pareto optimal solutions")
    print("\nYour actions:", actions)
    print(pareto_sim)

def main():
    human=False; test_iters=1
    pareto_front, pareto_rew_matrix = read_paretos("Pipeline/data/3-objective.json")
    app = App.Approximator()
    con = Con.Conciliator(objectives=['Treasure','Time','Fuel'], R = app.mean)
    priority, preference = con.priority, con.preference
    run(con, app, human, preference, test_iters, pareto_front, pareto_rew_matrix)

if __name__ == "__main__":
    main()