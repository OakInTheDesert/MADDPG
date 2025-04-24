import torch
import os
import logging
from network import DDPG
from training import PrioritizedReplayBuffer


class Agent:
    def __init__(self, n_states, n_actions, hidden_size, agent, batch_size, alpha, epsilon, gamma, tau, device, buffer_size, is_training):
        self.agent = agent
        self.net = DDPG(n_states, n_actions, hidden_size, batch_size, alpha, epsilon, gamma, tau, device)
        # Check existing model file
        self.actor_net_path = f"./models/{agent}_actor_net.pth"
        self.critic_net_path = f"./models/{agent}_critic_net.pth"
        if os.path.exists(self.actor_net_path) and os.path.exists(self.critic_net_path):
            try:
                actor_net_state_dict = torch.load(self.actor_net_path, map_location=device)
                critic_net_state_dict = torch.load(self.critic_net_path, map_location=device)
                self.net.actor_net.load_state_dict(actor_net_state_dict)
                self.net.target_actor_net.load_state_dict(actor_net_state_dict)
                self.net.critic_net.load_state_dict(critic_net_state_dict)
                self.net.target_critic_net.load_state_dict(critic_net_state_dict)
            except Exception as e:
                logging.error(f"Load {agent} model file error: {e}")
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        self.episode_rewards = {}
        self.avg_episode_rewards = []
        self.actor_net_history = {}
        self.critic_net_history = {}
        self.actor_iteration_losses = []
        self.critic_iteration_losses = []
        self.is_training = is_training
        self.actor_save_path = f"./models/{agent}_actor_net.pth"
        self.critic_save_path = f"./models/{agent}_critic_net.pth"

    def record_net(self, version):
        self.actor_net_history[version] = self.net.get_net()['actor_net']
        self.critic_net_history[version] = self.net.get_net()['critic_net']
        torch.save(self.actor_net_history[version], f"./models/{self.agent}_actor_net_{version}.pth")
        torch.save(self.critic_net_history[version], f"./models/{self.agent}_critic_net_{version}.pth")

    def train(self, iteration, other_agent, writer):
        self.net.train(self.replay_buffer, self.actor_iteration_losses, self.critic_iteration_losses, self.agent, other_agent, self.is_training)
        writer.add_scalar('{}_actor_iteration_losses'.format(self.agent), self.actor_iteration_losses[-1], iteration)
        writer.add_scalar('{}_critic_iteration_losses'.format(self.agent), self.critic_iteration_losses[-1], iteration)

    def save(self):
        avg_dict = {k: sum(v) / len(v) for k, v in self.episode_rewards.items() if v}
        self.avg_episode_rewards = list(avg_dict.values())
        max_key = max(avg_dict, key=avg_dict.get)
        torch.save(self.actor_net_history[max_key], self.actor_save_path)
        torch.save(self.critic_net_history[max_key], self.critic_save_path)
