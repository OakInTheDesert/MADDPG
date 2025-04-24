import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
import random
import logging
from env import AdversarialEnv
from network import DDPG


# Experience replay buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action1, action2, reward1, reward2, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)
        self.buffer[self.position] = (state, action1, action2, reward1, reward2, next_state, done)
        self.priorities[self.position] = max_priority  # Assign max priority initially
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Calculate the sampling probabilities (using priorities)
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling weight
        total_samples = len(self.buffer)
        weights = (total_samples * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def reset(self):
        self.buffer = []
        self.priorities = []

    def __len__(self):
        return len(self.buffer)


class SampleGenerator:
    def __init__(self, attacker, defender, n_states, n_actions, hidden_size, buffer_size, epsilon, epsilon_min, epsilon_decay):
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.attacker = attacker
        self.defender = defender

        self.total_episodes = 0
        self.num_W = 0
        self.num_L = 0
        self.num_T = 0

        # Experience queue setup
        self.experience_queue = Queue(maxsize=buffer_size)
        self.stats_queue = Queue()

        self.num_workers = max(mp.cpu_count() - 1, 1)
        logging.info(f"Parallel processes: {self.num_workers}")

        manager = Manager()
        self.shared_params = manager.dict()
        self.shared_params['version'] = 0
        self.shared_params['attacker_net'] = attacker.net.get_net()
        self.shared_params['defender_net'] = defender.net.get_net()
        self.shared_params['epsilon'] = epsilon

        self.workers = []

    def generate(self):
        for i in range(self.num_workers):
            p = Process(target=self.sample_worker, args=(i, self.n_states, self.n_actions, self.hidden_size))
            p.daemon = True
            p.start()
            self.workers.append(p)

    def sample_worker(self, worker_id, n_states, n_actions, hidden_size):
        env = AdversarialEnv()

        attacker_net = DDPG(n_states, n_actions, hidden_size)
        defender_net = DDPG(n_states, n_actions, hidden_size)

        local_version = self.shared_params.get('version', 0)
        attacker_net.load_params(self.shared_params['attacker_net'])
        defender_net.load_params(self.shared_params['defender_net'])
        epsilon = self.shared_params.get('epsilon', 0.8)
        attacker_net.set_epsilon(epsilon)
        defender_net.set_epsilon(epsilon)

        while True:
            state = env.reset()
            episode_reward1 = 0
            episode_reward2 = 0
            steps = 0
            done = False
            shared_version = self.shared_params.get('version', 0)
            if shared_version > local_version:
                local_version = shared_version
                attacker_net.load_params(self.shared_params['attacker_net'])
                defender_net.load_params(self.shared_params['defender_net'])
                epsilon = self.shared_params.get('epsilon', 0.8)
                attacker_net.set_epsilon(epsilon)
                defender_net.set_epsilon(epsilon)
                logging.info(f"Worker {worker_id} updated parameters to version {local_version}.")
            while not done:
                action1 = attacker_net.evaluate(state)
                action2 = defender_net.evaluate(state)
                next_state, reward1, reward2, done, _ = env.step(action1, action2)
                shared_version = self.shared_params.get('version', 0)
                if (random.random() < 0.2 and not self.experience_queue.full() or done) and not shared_version > local_version:
                    experience = (state, action1, action2, reward1, reward2, next_state, done)
                    self.experience_queue.put(experience)
                state = next_state
                episode_reward1 += reward1
                episode_reward2 += reward2
                steps += 1
                env.plt_counter = steps
                if steps >= 150 / env.time_step:
                    done = True
            stat = {'worker_id': worker_id, 'episode_reward1': episode_reward1, 'episode_reward2': episode_reward2, 'version': local_version, 'result': env.result, 'steps': steps}
            self.stats_queue.put(stat)
            logging.info(f"Worker {worker_id} finished an episode with result: {env.result}")

    def update_shared_params(self, version, attacker_net, defender_net):
        self.shared_params['version'] = version
        self.shared_params['attacker_net'] = attacker_net
        self.shared_params['defender_net'] = defender_net
        epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon = epsilon
        self.shared_params['epsilon'] = epsilon

    def transfer_experience_replay(self):
        try:
            exp = self.experience_queue.get(timeout=1)
            self.attacker.replay_buffer.push(*exp)
            self.defender.replay_buffer.push(*exp)
        except:
            return -1
        return 0

    def compile_results(self, writer):
        try:
            stat = self.stats_queue.get(timeout=1)
            self.total_episodes += 1
            if stat['version'] not in self.attacker.episode_rewards:
                self.attacker.episode_rewards[stat['version']] = []
            attacker_episode_reward = stat['episode_reward1']
            self.attacker.episode_rewards[stat['version']].append(attacker_episode_reward)
            writer.add_scalar('attacker_version_reward', sum(self.attacker.episode_rewards[stat['version']]) / len(self.attacker.episode_rewards[stat['version']]), stat['version'])
            if stat['version'] not in self.defender.episode_rewards:
                self.defender.episode_rewards[stat['version']] = []
            defender_episode_reward = stat['episode_reward2']
            self.defender.episode_rewards[stat['version']].append(defender_episode_reward)
            writer.add_scalar('defender_version_reward', sum(self.defender.episode_rewards[stat['version']]) / len(self.defender.episode_rewards[stat['version']]), stat['version'])
            if stat['result'] == "Win":
                self.num_W += 1
            elif stat['result'] == "Loss":
                self.num_L += 1
            else:
                self.num_T += 1
        except:
            return -1
        return 0
