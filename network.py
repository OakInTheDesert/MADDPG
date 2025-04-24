import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc4(x))
        return x


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class DDPG:
    def __init__(self, n_states, n_actions, hidden_size, batch_size=None, alpha=None, epsilon=None, gamma=None, tau=None, device='cpu'):
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        self.actor_net = ActorNetwork(n_states, hidden_size, n_actions).to(device)
        self.critic_net = CriticNetwork(n_states + n_actions * 2, hidden_size, 1).to(device)
        self.target_actor_net = ActorNetwork(n_states, hidden_size, n_actions).to(device)
        self.target_critic_net = CriticNetwork(n_states + n_actions * 2, hidden_size, 1).to(device)

        if alpha is not None:
            self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=alpha)
            self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=alpha)
        else:
            self.actor_optimizer = None
            self.critic_optimizer = None

        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.target_actor_net.eval()
        self.target_critic_net.eval()

    def load_params(self, params):
        self.actor_net.load_state_dict(params['actor_net'])
        self.critic_net.load_state_dict(params['critic_net'])
        
    def get_net(self):
        return {
            'actor_net': {k: v.cpu() for k, v in self.actor_net.state_dict().items()},
            'critic_net': {k: v.cpu() for k, v in self.critic_net.state_dict().items()}
        }

    def train(self, replay_buffer, actor_iteration_losses, critic_iteration_losses, agent_name, other_agent, is_training):
        self.actor_net.train()
        self.critic_net.train()

        batch, indices, weights = replay_buffer.sample(self.batch_size)
        if agent_name == 'attacker':
            state_batch, action1_batch, action2_batch, reward_batch, _, next_state_batch, done_batch, weights = self._transfer_data(batch, weights)
        else:
            state_batch, action1_batch, action2_batch, _, reward_batch, next_state_batch, done_batch, weights = self._transfer_data(batch, weights)

        with torch.no_grad():
            if agent_name == 'attacker':
                next_state_actions1 = self.target_actor_net(next_state_batch)
                next_state_actions2 = other_agent.net.target_actor_net(next_state_batch)
            else:
                next_state_actions1 = other_agent.net.target_actor_net(next_state_batch)
                next_state_actions2 = self.target_actor_net(next_state_batch)

            target_q_inputs = torch.cat([next_state_batch, next_state_actions1, next_state_actions2], dim=1)
            next_state_values = self.target_critic_net(target_q_inputs).squeeze()
            target_q_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)

        predicted_q_values = self.critic_net(torch.cat([state_batch, action1_batch, action2_batch], dim=1)).squeeze()

        critic_loss = (weights * (predicted_q_values - target_q_values).pow(2)).mean()
        
        if is_training:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        
        if agent_name == 'attacker':
            actions1 = self.actor_net(state_batch)
            reg = (actions1 ** 2).mean() * 1e-3
            with torch.no_grad():
                actions2 = other_agent.net.actor_net(state_batch)
        else:
            with torch.no_grad():
                actions1 = other_agent.net.actor_net(state_batch)
            actions2 = self.actor_net(state_batch)
            reg = (actions2 ** 2).mean() * 1e-3

        q_inputs = torch.cat([state_batch, actions1, actions2], dim=1)
        with torch.no_grad():
            q_values = self.critic_net(q_inputs).squeeze()
        
        actor_loss = (weights * (-q_values)).mean() + reg

        if is_training:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
        td_errors = (predicted_q_values - target_q_values).abs().detach().cpu().numpy()
        replay_buffer.update_priorities(indices, td_errors)

        actor_iteration_losses.append(actor_loss.item())
        critic_iteration_losses.append(critic_loss.item())

    def evaluate(self, state):
        self.actor_net.eval()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            if np.random.random() < self.epsilon:
                action = np.random.uniform(-1, 1, size=self.n_actions)
            else:
                action = self.actor_net(state_tensor.unsqueeze(0)).item()
        return action
    
    def _transfer_data(self, batch, weights):
        state_batch, action_batch1, action_batch2, reward_batch1, reward_batch2, next_state_batch, done_batch = zip(*batch)
        state_batch = np.array(state_batch)
        action_batch1 = np.array(action_batch1)
        action_batch2 = np.array(action_batch2)
        reward_batch1 = np.array(reward_batch1)
        reward_batch2 = np.array(reward_batch2)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)

        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        action_batch1 = torch.tensor(action_batch1, dtype=torch.float32, device=self.device).unsqueeze(1)
        reward_batch1 = torch.tensor(reward_batch1, dtype=torch.float32, device=self.device)
        action_batch2 = torch.tensor(action_batch2, dtype=torch.float32, device=self.device).unsqueeze(1)
        reward_batch2 = torch.tensor(reward_batch2, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.int64, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return state_batch, action_batch1, action_batch2, reward_batch1, reward_batch2, next_state_batch, done_batch, weights

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    # Update the target network
    def soft_update(self):
        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        torch.save(self.actor_net.state_dict(), path)
        torch.save(self.critic_net.state_dict(), path)
