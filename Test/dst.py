import numpy as np
import time
import deep_sea_treasure
from deep_sea_treasure import DeepSeaTreasureV0
import pygame
import pandas as pd
from deep_sea_treasure import FuelWrapper
import conciliator_v2 as con 
import approximator_v1 as app
    
def init_dst():
    # Make sure experiment are reproducible, so people can use the exact same versions
    print(f"Using DST {deep_sea_treasure.__version__.VERSION} ({deep_sea_treasure.__version__.COMMIT_HASH})")

    dst: DeepSeaTreasureV0 =  FuelWrapper.new(DeepSeaTreasureV0.new(
        max_steps=1000,
        render_treasure_values=True,
    ))
    return dst

def run(human, priority, preference):
    dst = init_dst()

    dst.render()

    stop: bool = False
    time_reward: int = 0

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
            action = next_con_action(preference)

        _, reward, done, debug_info = dst.step(action)
        time_reward += int(reward[1])
        
        if done:
            received_rews = np.asarray([int(reward[0]), time_reward, int(reward[2])])
            print_results(received=received_rews, preferred=preference)
            time_reward = 0
        
        if not stop:
                dst.render()
                time.sleep(0.25)

        if done:
            dst.reset()
    
    print(f"Conciliator steering has ended. Bye!")

def print_results(received, preferred):
    labels = ["treasure", "time", "fuel"]
    d = {'received': received, 'preferred': preferred, 'difference': received-preferred}
    df = pd.DataFrame(data=d, index=labels)
    print(df)

def main():
    conc = con.Conciliator(eps=1e-16, R = np.array([30,1000,18]))
    #appr = app()
    priority, preference = conc.priority, conc.preference
    print(priority)
    print(preference)
    human=True; run_test=False
    thresh = 10e-2; train_iters = 10e+2; test_iters = 10
    if human:
        run(human, priority, preference)
    if run_test:
        run(run_test, priority, preference)

if __name__ == "__main__":
    main()