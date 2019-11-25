import retro
import envs
import numpy as np
import time
game='StreetFighterIISpecialChampionEdition-Genesis'
env = retro.make(game=game)
# Previous env es el wrapper base de SFII
env = envs.ButtonsRemapper(env, game=game)
# Previous env es el wrapper base de SFII
reward_new = True
if reward_new:
    env = envs.EnvStreetFighterII(env)

while True:
    state = env.reset()
    done = False
    reward_total = 0.0
    step = 0
    while not done:
        action = np.random.choice(range(env.action_space.n))
        next_state, reward, done, info = env.step(action)
        reward_total += reward
        env.render()
        if reward_new and reward != 0:
            print("Action:", envs.ACTIONS_CONFIG["games"][game]["actions"][action], "Observation:", next_state.shape, "Reward:", reward, "Done:", done, "Info:", info)
            time.sleep(abs(reward)*0.3)
        elif reward >= 1000:
            print("Action:", envs.ACTIONS_CONFIG["games"][game]["actions"][action], "Observation:", next_state.shape, "Reward:", reward, "Done:", done, "Info:", info)
            time.sleep(abs(reward) / 300)
        state = next_state
        step = step + 1
    print("Episode reward", reward_total)
    print("Episode steps", step)