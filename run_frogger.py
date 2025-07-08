import argparse
import time
import pygame
from frogger_env import FroggerEnv, Action

def run_human(env: FroggerEnv):
    obs, _ = env.reset()
    env.render()  # Fenster initialisieren

    while True:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = Action.UP.value
                elif event.key == pygame.K_DOWN:
                    action = Action.DOWN.value
                elif event.key == pygame.K_LEFT:
                    action = Action.LEFT.value
                elif event.key == pygame.K_RIGHT:
                    action = Action.RIGHT.value
                elif event.key == pygame.K_SPACE:
                    action = Action.STAY.value


        if action is not None:
            obs, reward, done, _, _ = env.step(action)

            print(f"Action: {action}, Reward: {reward:.2f}")
            print(f"Obs: {obs}")

            env.render()
            if done:
                obs, _ = env.reset()
                env.render()

def run_random(env: FroggerEnv):
    obs, _ = env.reset()

    env.render()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)

        print(f"Action: {action}, Reward: {reward:.2f}")
        print(f"Obs: {obs}")

        env.render()
        time.sleep(0.1)
        if done:
            obs, _ = env.reset()
            env.render()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["human", "random"], default="random")
    args = parser.parse_args()

    env = FroggerEnv(render_mode="human")

    if args.agent == "human":
        run_human(env)
    else:
        run_random(env)
