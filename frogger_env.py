import pygame
import random
import numpy as np
import math
from enum import Enum
import gymnasium as gym
from gymnasium import spaces

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4  

# Parameters
NUM_LANES = 8
GRID_WIDTH = 15
GRID_HEIGHT = NUM_LANES + 3
TILE_SIZE = 30
WINDOW_WIDTH = GRID_WIDTH * TILE_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * TILE_SIZE
MAX_CARS = 10
CAR_SPAWN_INTERVAL_RANGE = (0, 2)
FAST_LANE_SPEED = 1.0
SLOW_LANE_SPEED = 0.5
FPS = 4 # ``GridWorldEnv``  modes “rgb_array” and “human”  render at 4 FPS.

# Colors
COLOR_START = (0, 0, 255)
COLOR_SAFE = (150, 75, 0)
COLOR_GOAL = (0, 128, 0)
COLOR_LANE = (160, 160, 160)
COLOR_GRID = (50, 50, 50)
COLOR_FROG = (0, 255, 0)
COLOR_CAR = (255, 0, 0)
COLOR_BACKGROUND = (255, 255, 255)


class FroggerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0] + [0.0, 0.0, -1.0] * MAX_CARS, dtype=np.float32), 
            high=np.array([1.0, 1.0] + [1.0, 1.0, 1.0] * MAX_CARS, dtype=np.float32), 
            shape=(2 + 3 * MAX_CARS,),
            dtype=np.float32
        )

        self.frog_pos = None
        self.cars = []
        self.spawn_timer = None 
        self.farthest_row = None
        self.lane_speeds = self.init_lane_speeds()
        self.lane_rows = list(self.lane_speeds.keys())

    def init_lane_speeds(self):
        speeds = {}
        m = NUM_LANES // 2
        k = math.ceil(m / 2)
        for i in range(m):
            speed = FAST_LANE_SPEED if i < k else SLOW_LANE_SPEED
            speeds[i + 1] = (1, speed)
        for i in range(m):
            speed = FAST_LANE_SPEED if i < k else SLOW_LANE_SPEED
            speeds[m + i + 2] = (-1, speed)
        return speeds

    def _get_obs(self):
        obs = [self.frog_pos[0] / GRID_HEIGHT, self.frog_pos[1] / GRID_WIDTH]
        for car in self.cars:
            obs.extend([
                car['row'] / GRID_HEIGHT,
                car['col'] / GRID_WIDTH,
                car['dir'] * car['speed'] / FAST_LANE_SPEED
            ])
        while len(obs) < 2 + 3 * MAX_CARS:
            obs.extend([0, 0, 0])
        return np.array(obs[:2 + 3 * MAX_CARS], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.frog_pos = [GRID_HEIGHT - 1, GRID_WIDTH // 2]
        self.cars = []
        self.spawn_timer = random.randint(*CAR_SPAWN_INTERVAL_RANGE)
        self.farthest_row = self.frog_pos[0]

        num_initial_cars = random.randint(int(0.5 * MAX_CARS), int(0.8 * MAX_CARS))
        for _ in range(num_initial_cars):
            self.spawn_car()

        return self._get_obs(), {}

    def spawn_car(self):
        if len(self.cars) >= MAX_CARS:
            return
        lane = random.choice(self.lane_rows)
        direction, speed = self.lane_speeds.get(lane, (0, 0))
        for car in self.cars:
            if car['row'] == lane:
                if direction == 1 and car['col'] < 2:
                    return
                elif direction == -1 and car['col'] > GRID_WIDTH - 3:
                    return
        col = 0 if direction == 1 else GRID_WIDTH - 1
        self.cars.append({'row': lane, 'col': float(col), 'dir': direction, 'speed': speed})

    def reward(self, collided: bool, reached_goal: bool, advanced_row: bool) -> float:
        if collided:
            return -1.0
        elif reached_goal:
            return 1.0
        reward = -0.01
        if advanced_row: # o/w can reward loops
            reward += 0.1
        return reward

    def step(self, action):
        if action == Action.UP.value and self.frog_pos[0] > 0:
            self.frog_pos[0] -= 1
        elif action == Action.DOWN.value and self.frog_pos[0] < GRID_HEIGHT - 1:
            self.frog_pos[0] += 1
        elif action == Action.LEFT.value and self.frog_pos[1] > 0:
            self.frog_pos[1] -= 1
        elif action == Action.RIGHT.value and self.frog_pos[1] < GRID_WIDTH - 1:
            self.frog_pos[1] += 1
        elif action == Action.STAY.value:
            pass

        for car in self.cars:
            car['col'] += car['dir'] * car['speed']

        self.cars = [car for car in self.cars if 0 <= car['col'] < GRID_WIDTH]
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_car()
            self.spawn_timer = random.randint(*CAR_SPAWN_INTERVAL_RANGE)

        advanced_row = self.frog_pos[0] < self.farthest_row
        if advanced_row:
            self.farthest_row = self.frog_pos[0]

        collided = False
        reached_goal = False

        frog_rect = pygame.Rect(self.frog_pos[1] * TILE_SIZE, self.frog_pos[0] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        for car in self.cars:
            car_rect = pygame.Rect(car['col'] * TILE_SIZE, car['row'] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if frog_rect.colliderect(car_rect):
                collided = True
                break

        if not collided and self.frog_pos[0] == 0:
            reached_goal = True

        done = collided or reached_goal
        reward = self.reward(collided, reached_goal, advanced_row)

        return self._get_obs(), reward, done, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.window.fill(COLOR_BACKGROUND)

        for row in range(GRID_HEIGHT):
            y = row * TILE_SIZE
            if row == GRID_HEIGHT - 1:
                color = COLOR_START
            elif row == 0:
                color = COLOR_GOAL
            elif row == (NUM_LANES // 2 + 1):
                color = COLOR_SAFE
            elif 1 <= row <= (NUM_LANES + 1):
                color = COLOR_LANE
            else:
                color = COLOR_BACKGROUND
            pygame.draw.rect(self.window, color, (0, y, WINDOW_WIDTH, TILE_SIZE))

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.window, COLOR_GRID, rect, 1)

        for car in self.cars:
            car_rect = pygame.Rect(car['col'] * TILE_SIZE, car['row'] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(self.window, COLOR_CAR, car_rect)

        frog_rect = pygame.Rect(self.frog_pos[1] * TILE_SIZE, self.frog_pos[0] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(self.window, COLOR_FROG, frog_rect)

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
