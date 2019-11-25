import retro
import envs
import numpy as np
import time
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
# Previous env es el wrapper base de SFII
env = envs.ButtonsRemapper(env, game='StreetFighterIISpecialChampionEdition-Genesis')

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
        state = next_state
        step = step + 1
    print("Episode reward", reward_total)
    print("Episode steps", step)