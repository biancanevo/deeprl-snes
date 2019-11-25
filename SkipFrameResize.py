import retro
import envs
import numpy as np
import time
game='StreetFighterIISpecialChampionEdition-Genesis'
env = retro.make(game=game)
# Previous env es el wrapper base de SFII
env = envs.ButtonsRemapper(env, game=game)
# Previous env es el wrapper base de SFII
env = envs.EnvStreetFighterII(env)
env = envs.SkipFrames(env)
togray = False
env = envs.WarpFrame(env, togray=togray)
from gym.envs.classic_control.rendering import SimpleImageViewer
viewer = SimpleImageViewer()
while True:
    ims = []
    state = env.reset()
    done = False
    reward_total = 0.0
    step = 0
    while not done:
        action = np.random.choice(range(env.action_space.n))
        next_state, reward, done, info = env.step(action)
        reward_total += reward
        viewer.imshow(next_state if not togray else np.tile(next_state, (1, 3)))
        if reward != 0:
            print("Action:", envs.ACTIONS_CONFIG["games"][game]["actions"][action], "Observation:", next_state.shape, "Reward:", reward, "Done:", done, "Info:", info)
            time.sleep(abs(reward)*0.3)
        state = next_state
        step = step + 1
    print("Episode reward", reward_total)
    print("Episode steps", step)