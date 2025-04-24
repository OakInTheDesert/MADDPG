from env import AdversarialEnv
from network import Network
import numpy as np
import torch

if __name__ == '__main__':
    # Basic script to confirm code is working properly
    env = AdversarialEnv()
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attacker_model = Network(3 * n_states, hidden_size, n_actions).to(device)
    attacker_model.load_state_dict(torch.load('./models/attacker_value_net.pth'))
    attacker_model.eval()

    defender_model = Network(3 * n_states, hidden_size, n_actions).to(device)
    defender_model.load_state_dict(torch.load('./models/defender_value_net.pth'))
    defender_model.eval()

    for episode in range(1):
        obs = env.reset()
        step = 0
        while step <= 150 / env.time_step:
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # action1 = attacker_model(state_tensor).argmax(dim=1).item()
                action2 = defender_model(state_tensor).argmax(dim=1).item()

            # action1 = np.random.randint(0, 10)
            # action2 = np.random.randint(0, 10)

            action1 = 5

            obs, reward1, reward2, done, _ = env.step(action1, action2, True)
            env.render()

            if done:
                break

            step += 1
    input("Press Enter to quit...")
