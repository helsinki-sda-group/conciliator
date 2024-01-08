import numpy as np
import time
import deep_sea_treasure
import sys
from deep_sea_treasure import DeepSeaTreasureV0
import pygame
import pandas as pd
from deep_sea_treasure import FuelWrapper
import conciliator_v2 as Con 
import approximator_v1 as App
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
    print(f"Using DST {deep_sea_treasure.__version__.VERSION} ({deep_sea_treasure.__version__.COMMIT_HASH})")

    dst: DeepSeaTreasureV0 =  FuelWrapper.new(DeepSeaTreasureV0.new(
        max_steps=100,
        render_treasure_values=True,
    ))
    return dst

def run(con, app, priority, preference, users=1, human=True):
    dst = init_dst()

    actions = []
    iters = 0

    stop: bool = False
    time_reward: int = 0
    current_state = None        
    running_reward = [0,0,0]

    while not stop:
        if current_state is None:
            current_state = np.zeros((2,11))
            current_state[:,1:] = np.column_stack(tuple(dst.treasures.keys()))
        if human:
            dst.render()
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
            action = app.next_action(preference,current_state,dst, running_reward)

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
        running_reward += [reward[0],time_reward,reward[2]]
        
        if done:
            received_rews = np.asarray([int(reward[0]), time_reward+100, int(reward[2])+30])
            print_results(received=received_rews, preferred=preference, actions=actions)
            iters += 1
            time_reward = 0
        
        if not stop:
            if human:
                dst.render()
                time.sleep(0.25)

        if done:
            iters += 1
            dst.reset()
            if iters >= users:
                stop = True
    
    print(f"Conciliator steering has ended. Bye!\n\n")

def print_results(received, preferred, actions):
    print("\nActions:", actions)
    pareto_front, pareto_rew_matrix = read_paretos("Pipeline/3-objective.json")
    labels = ["treasure", "time", "fuel"]
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
    pareto_sim["% of same actions"] = metrics
    print("\n\nDifference to Pareto optimal solutions")
    print("\nYour actions:", actions)
    print(pareto_sim)

def main():
    # TODO:
    # Training
    # tracker = OfflineEmissionsTracker(country_iso_code="FIN")
    # tracker.start()
    app = App.Approximator()
    con = Con.Conciliator(objectives=['Treasure','Time','Fuel'], R = app.mean)
    # train_emissions = tracker.stop()*1000
    priority, preference = con.priority, con.preference
    print("Pref:",preference)
    run(con, app, priority, preference, users=1, human=False)

    # Testing
    # Priority profiles
    # 1. "eco": 1/9 time, 2/9 treasure and 6/9 fuel
    # 2. "treasure rush": 8/16 time, 7/16 treasure and 1/16 fuel
    # 3. "rush": 2/4 time, 1/4 treasure and 1/4 fuel
    # 4. "balanced": 1/3 time, 1/3 treasure and 1/3 fuel
    # priorities = [[1/9,2/9,6/9],[8/16,7/16,1/16],[2/4,1/4,1/4],[1/3,1/3,1/3]]
    # tracker.start()
    # i = 0
    # for priority in priorities:
    #     print(f"Profile {i}: {priority}")
    #     priority, preference = con.priority, con.preference
    #     # ESR block
    #     # Seek out 1 and calculate 3 policies for each priority profile 
    #     print(f"\nESR tests\n")
    #     run(con, app, human, priority, preference, users=1)
    #     fitted_weights = con.scalarisation_fit()
    #     esr_policies = app.policy_fit(n_pol=3, cond="ESR", weights=fitted_weights)
    #     for policy in esr_policies:
    #         print_results(preference, priority, policy)
    #     # SER block
    #     # Calculate an averaged set of 10 policies for each priority profile
    #     print(f"\nSER tests\n")
    #     ser_policy = app.policy_fit(n_pol=10, cond="SER", weights=fitted_weights)
    #     print_results(preference, priority, ser_policy)
    #     i += 1
    # test_emissions = tracker.stop()*1000

    # Emissions
    # print(f"Training emissions: {train_emissions} g of CO2")
    # print(f"Testing emissions: {test_emissions} g of CO2")
    # print(f"Total emissions: {train_emissions+test_emissions} g of CO2")


if __name__ == "__main__":
    main()