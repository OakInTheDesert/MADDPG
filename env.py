import numpy as np
import matplotlib.pyplot as plt
from gym import Env
from gym.spaces import Box, Discrete
import pymap3d as pm
import logging


class AdversarialEnv(Env):
    def __init__(self):
        self.acc_min = -1
        self.acc_max = 1

        # self.action_space = Box(low=self.acc_min, high=self.acc_max, shape=(1,), dtype=np.float32)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(9,))

        self.attacker = None
        self.defender = None
        self.target = None

        self.done = False
        self.result = "Tie"

        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.z_min = None
        self.z_max = None

        self.effective_radius = 0.5

        self.time_step = 0.1

        self.fig = None
        self.ax = None
        self.attacker_trajectory = []
        self.defender_trajectory = []
        self.plt_counter = 0

    def reset(self):
        self.attacker = Attacker()
        self.defender = Defender()
        self.target = Target()

        self.done = False

        self.x_min = min(self.attacker.pos[0], self.defender.pos[0], self.target.pos[0]) - 100
        self.x_max = max(self.attacker.pos[0], self.defender.pos[0], self.target.pos[0]) + 100
        self.y_min = min(self.attacker.pos[1], self.defender.pos[1], self.target.pos[1]) - 100
        self.y_max = max(self.attacker.pos[1], self.defender.pos[1], self.target.pos[1]) + 100
        self.z_min = min(self.attacker.pos[2], self.defender.pos[2], self.target.pos[2]) - 100
        self.z_max = max(self.attacker.pos[2], self.defender.pos[2], self.target.pos[2]) + 100

        attacker_direction = self.target.pos - self.attacker.pos
        attacker_direction = attacker_direction / np.linalg.norm(attacker_direction)
        self.attacker.vel = 3.4 * attacker_direction

        defender_direction = self.attacker.pos - self.defender.pos
        defender_direction = defender_direction / np.linalg.norm(defender_direction)
        self.defender.vel = 0.1 * defender_direction

        self.state = np.array([self.attacker.pos[0], self.attacker.pos[1], self.attacker.pos[2], self.defender.pos[0], self.defender.pos[1], self.defender.pos[2], self.target.pos[0], self.target.pos[1], self.target.pos[2]])

        return self.state

    def step(self, action1, action2, verbose=False):
        self.attacker.step(action1, self.time_step)
        self.defender.step(action2, self.time_step)

        att_def_distance = self.calc_distance(self.attacker.pos, self.defender.pos)
        att_tar_distance = self.calc_distance(self.attacker.pos, self.target.pos)
        def_tar_distance = self.calc_distance(self.defender.pos, self.target.pos)

        if verbose:
            print(self.plt_counter)
            print(att_def_distance, att_tar_distance, def_tar_distance)

        # Get rewards for attacker and defender
        attacker_reward = self.attacker.reward_function(self.defender.pos, self.target.pos, att_def_distance, att_tar_distance, self.effective_radius)
        defender_reward = self.defender.reward_function(self.attacker.pos, self.target.pos, att_def_distance, att_tar_distance, def_tar_distance, self.effective_radius)

        if att_tar_distance < self.effective_radius or def_tar_distance < self.effective_radius or self.defender.lla_pos[2] <= 0:
            self.done = True
            self.result = "Win"
        elif att_def_distance < self.effective_radius or self.attacker.lla_pos[2] <= 0:
            self.done = True
            self.result = "Loss"

        self.state = np.array([self.attacker.pos[0], self.attacker.pos[1], self.attacker.pos[2], self.defender.pos[0], self.defender.pos[1], self.defender.pos[2], self.target.pos[0], self.target.pos[1], self.target.pos[2]])

        return self.state, attacker_reward, defender_reward, self.done, None

    def render(self):
        # This creates a single frame
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim([self.x_min, self.x_max])
            self.ax.set_ylim([self.y_min, self.y_max])
            self.ax.set_zlim([self.z_min, self.z_max])
            plt.ion()

        self.attacker_trajectory.append(self.attacker.pos)
        self.defender_trajectory.append(self.defender.pos)

        user_xlim = self.ax.get_xlim3d()
        user_ylim = self.ax.get_ylim3d()
        user_zlim = self.ax.get_zlim3d()
        self.ax.cla()
        self.ax.set_xlim3d(user_xlim)
        self.ax.set_ylim3d(user_ylim)
        self.ax.set_zlim3d(user_zlim)

        self.ax.scatter(self.target.pos[0], self.target.pos[1], self.target.pos[2], c='g', marker='o', label='Target')
        self.ax.scatter(self.attacker.pos[0], self.attacker.pos[1], self.attacker.pos[2], c='r', marker='^', label='Attacker')
        self.ax.scatter(self.defender.pos[0], self.defender.pos[1], self.defender.pos[2], c='b', marker='s', label='Defender')

        if len(self.attacker_trajectory) > 1:
            attacker_trajectory = np.array(self.attacker_trajectory)
            defender_trajectory = np.array(self.defender_trajectory)
            self.ax.plot(attacker_trajectory[:, 0], attacker_trajectory[:, 1], attacker_trajectory[:, 2], 'r-', alpha=0.5)
            self.ax.plot(defender_trajectory[:, 0], defender_trajectory[:, 1], defender_trajectory[:, 2], 'b-', alpha=0.5)

        self.ax.legend()
        self.ax.set_title(f"Step: {self.plt_counter}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

        self.plt_counter += 1

    @staticmethod
    def calc_distance(pos1, pos2):
        x = abs(pos1[0] - pos2[0])
        y = abs(pos1[1] - pos2[1])
        z = abs(pos1[2] - pos2[2])
        return np.sqrt(x ** 2 + y ** 2 + z ** 2)


class Attacker:
    def __init__(self):
        self.lla_pos = np.array([26.07, 119.30, 60])
        self.pos = np.array(pm.geodetic2ecef(self.lla_pos[0], self.lla_pos[1], self.lla_pos[2] * 1000)) / 1000
        self.vel = np.array([0, 0, 0])

    def step(self, action, time_step):
        acc = action

        # update velocity
        speed = np.linalg.norm(self.vel)
        if speed > 0:
            vertical_vec = self.pos / np.linalg.norm(self.pos)
            horizontal_vec = np.cross(self.vel, vertical_vec)
            horizontal_vec = horizontal_vec / np.linalg.norm(horizontal_vec)

            direction = self.vel + acc / 1000 * time_step * horizontal_vec * 10
            direction = direction / np.linalg.norm(direction)

            new_speed = max(0, speed - 0.01 * time_step)

            self.vel = direction * new_speed
            self.pos = self.pos + self.vel * time_step
            self.lla_pos = np.array([*pm.ecef2geodetic(self.pos[0] * 1000, self.pos[1] * 1000, self.pos[2] * 1000)[:2], pm.ecef2geodetic(self.pos[0] * 1000, self.pos[1] * 1000, self.pos[2] * 1000)[2] / 1000])

    def reward_function(self, defender, target, att_def_distance, att_tar_distance, effective_radius):
        # We define our reward function to be the inverse distance between the agent and its goal/adversary
        reward = 0
        eps = 0.5

        if att_def_distance <= effective_radius:
            reward -= 1000
            logging.info('Reached Terminal State, the Defender intercepted the Attacker!')
        elif att_def_distance <= effective_radius * 5:
            reward -= 100
        elif att_def_distance <= effective_radius * 10:
            reward -= 50
        reward -= 250 / (att_def_distance + eps)

        if att_tar_distance <= effective_radius:
            reward += 2000
            logging.info('Reached Terminal State, the Attacker hit the Goal!')
        elif self.lla_pos[2] > 0 and att_tar_distance <= effective_radius * 5:
            reward += 200
        elif self.lla_pos[2] > 0 and att_tar_distance <= effective_radius * 10:
            reward += 100
        elif self.lla_pos[2] < 0:
            reward -= 1000
            logging.info('Reached Terminal State, the Attacker missed the Target!')
        reward += 500 / (att_tar_distance + eps)

        reward -= 1  # reward shaping

        return reward


class Defender:
    def __init__(self):
        self.lla_pos = np.array([25.23, 121.44, 3.87])
        self.pos = np.array(pm.geodetic2ecef(self.lla_pos[0], self.lla_pos[1], self.lla_pos[2] * 1000)) / 1000
        self.vel = np.array([0, 0, 0])

    def step(self, action, time_step):
        acc = action

        # update velocity
        speed = np.linalg.norm(self.vel)
        if speed > 0:
            vertical_vec = self.pos / np.linalg.norm(self.pos)
            horizontal_vec = np.cross(self.vel, vertical_vec)
            horizontal_vec = horizontal_vec / np.linalg.norm(horizontal_vec)

            direction = self.vel + acc / 1000 * time_step * horizontal_vec * 20
            direction = direction / np.linalg.norm(direction)

            new_speed = max(3.6, speed + 0.05 * time_step)

            self.vel = direction * new_speed
            self.pos = self.pos + self.vel * time_step
            self.lla_pos = np.array([*pm.ecef2geodetic(self.pos[0] * 1000, self.pos[1] * 1000, self.pos[2] * 1000)[:2], pm.ecef2geodetic(self.pos[0] * 1000, self.pos[1] * 1000, self.pos[2] * 1000)[2] / 1000])

    def reward_function(self, attacker, target, att_def_distance, att_tar_distance, def_tar_distance, effective_radius):
        reward = 0
        eps = 0.5

        if att_def_distance <= effective_radius:
            reward += 2000  # if the def reaches attacker
        elif att_def_distance <= effective_radius * 5:
            reward += 200
        elif att_def_distance <= effective_radius * 10:
            reward += 100
        reward += 500 / (att_def_distance + eps)

        if att_tar_distance <= effective_radius:
            reward -= 1000  # if attacker gets to goal, large negative reward
        elif att_tar_distance <= effective_radius * 5:
            reward -= 100
        elif att_tar_distance <= effective_radius * 10:
            reward -= 50
        reward -= 250 / (att_tar_distance + eps)

        if def_tar_distance <= effective_radius:
            reward -= 1000
            logging.info('Reached Terminal State, the Defender hit the Target!')
        reward -= 125 / (def_tar_distance + eps)

        if self.lla_pos[2] <= 0:
            reward -= 1000
            logging.info('Reached Terminal State, the Defender missed the Attacker!')

        reward -= 1  # reward shaping

        return reward


class Target:
    def __init__(self):
        self.lla_pos = np.array([25.03, 121.52, 0])
        self.pos = np.array(pm.geodetic2ecef(self.lla_pos[0], self.lla_pos[1], self.lla_pos[2] * 1000)) / 1000
