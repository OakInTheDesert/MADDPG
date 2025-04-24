import matplotlib.pyplot as plt
import logging
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Lock
from env import AdversarialEnv
from training import SampleGenerator
from agent import Agent


def plot_training_results(attacker, defender, generator):
    # Plot the episode rewards over time
    plt.figure(figsize=(6, 6))
    plt.plot(attacker.avg_episode_rewards)
    plt.xlabel("Model Version")
    plt.ylabel("Reward")
    plt.title("Attacker Reward per Version")
    plt.savefig('./images/attacker_reward_per_version.png')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(defender.avg_episode_rewards)
    plt.xlabel("Model Version")
    plt.ylabel("Reward")
    plt.title("Defender Reward per Version")
    plt.savefig('./images/defender_reward_per_version.png')
    plt.show()

    # Plot the episode losses over time
    plt.figure(figsize=(6, 6))
    plt.plot(attacker.iteration_losses)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Attacker Loss per Iteration")
    plt.savefig('./images/attacker_loss_per_iteration.png')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(defender.iteration_losses)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Defender Loss per Iteration")
    plt.savefig('./images/defender_loss_per_iteration.png')
    plt.show()

    # Plot the bar graph for game results
    plt.figure(figsize=(6, 6))
    labels = ['Attacker wins', 'Defender wins', 'Ties']
    values = [generator.num_W, generator.num_L, generator.num_T]
    colors = ['red', 'blue', 'gray']
    plt.bar(labels, values, color=colors)
    plt.title('Game Results')
    plt.xlabel('Result')
    plt.ylabel('Number of Games')
    plt.savefig('./images/game_results.png')
    plt.show()


if __name__ == '__main__':
    # Logging settings
    log_dir = r"./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    images_dir = r"./images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    models_dir = r"./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    log_path = os.path.join(log_dir, "app.log")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(log_path, 'w')])
    writer = SummaryWriter('/root/tf-logs')

    # DDPG parameters
    attempts = 1000000
    gamma = 0.95  # discount Factor
    alpha = 0.001  # learning rate
    epsilon = 0.8  # randomization
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 1024
    buffer_size = 20000
    hidden_size = 128
    tau = 0.1
    update_target_frequency = 500
    update_parameter_frequency = 10000
    switch_frequency = 20000
    refresh_buffer_frequency = 50000

    # Environment parameters
    temp_env = AdversarialEnv()
    n_states = temp_env.observation_space.shape[0]
    n_actions = temp_env.action_space.shape[0]
    del temp_env

    try:
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Training on device: {device}")

        # Agent setup
        attacker = Agent(n_states, n_actions, hidden_size, 'attacker', batch_size, alpha, epsilon, gamma, tau, device, buffer_size, True)
        defender = Agent(n_states, n_actions, hidden_size, 'defender', batch_size, alpha, epsilon, gamma, tau, device, buffer_size, False)

        generator = SampleGenerator(attacker, defender, n_states, n_actions, hidden_size, buffer_size, epsilon, epsilon_min, epsilon_decay)
        generator.generate()

        # Training loop
        while attacker.replay_buffer.__len__() < buffer_size or defender.replay_buffer.__len__() < buffer_size:
            generator.transfer_experience_replay()

        lock = Lock()
        for i in range(attempts):
            if i % 1000 == 0:
                print(f"Training iteration: {i}")

            count = 0
            while not generator.stats_queue.empty() and count < 1000:
                if generator.compile_results(writer) == -1:
                    break
                count += 1

            if i != 0 and i % update_target_frequency == 0:
                attacker.net.soft_update()
                defender.net.soft_update()

            if i % update_parameter_frequency == 0:
                attacker.record_net(i)
                defender.record_net(i)
                if i != 0:
                    with lock:
                        attacker_params = attacker.net.get_net()
                        defender_params = defender.net.get_net()
                        generator.update_shared_params(i, attacker_params, defender_params)
                        logging.info(f"Main process updated shared parameters. Version: {i}")
            if i != 0 and i % switch_frequency == 0:
                if i >= attempts / 2:
                    attacker.is_training = True
                    defender.is_training = True
                else:
                    attacker.is_training = not attacker.is_training
                    defender.is_training = not defender.is_training
            if i != 0 and i % refresh_buffer_frequency == 0:
                tmp_attacker_replay_buffer = attacker.replay_buffer.sample(buffer_size // 3)
                tmp_defender_replay_buffer = defender.replay_buffer.sample(buffer_size // 3)
                attacker.replay_buffer.reset()
                defender.replay_buffer.reset()
                for experience in tmp_attacker_replay_buffer:
                    attacker.replay_buffer.push(*experience)
                for experience in tmp_defender_replay_buffer:
                    defender.replay_buffer.push(*experience)
                while attacker.replay_buffer.__len__() < buffer_size or defender.replay_buffer.__len__() < buffer_size:
                    generator.transfer_experience_replay()

            attacker.train(i, defender, writer)
            defender.train(i, attacker, writer)

        # Save the models
        attacker.save()
        defender.save()

        writer.close()

        # Plot the training results
        plot_training_results(attacker, defender, generator)

    except Exception as e:
        logging.exception(e)

    input("Press Enter to quit...")
