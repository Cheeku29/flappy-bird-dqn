import argparse
from pickletools import optimize
import random
import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml  
import os
import argparse

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent: 
    def __init__(self, param_set):
        self.param_set = param_set

        with open("params.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]    

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{param_set}_log.txt")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{param_set}_model.pth")  

    def run(self, is_training=True, render=False):

        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            steps = 0

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            best_reward = float("-inf")


        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)

            episode_rewards = 0
            terminated = False  

            while not terminated and episode_rewards < self.reward_threshold:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.long).to(device)
                else:
                     with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).argmax(dim=1).to(device)

                next_state, reward, terminated, _, _ = env.step(action.item())

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                    state = next_state
                    episode_rewards += reward

            print(f"for episode={episode + 1} total reward={episode_rewards} & epsilon={epsilon}")

            if is_training:
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

                if episode_rewards > best_reward:
                    log_msg = f"New best reward: {episode_rewards} (previous best: {best_reward}) for episode {episode + 1}\n"
                    
                    with open(self.LOG_FILE, "a") as log_file:
                        log_file.write(log_msg)

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_rewards

            if is_training and len(memory) >= self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)

                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0
    

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        state, action, next_state, reward, terminated = zip(*mini_batch)
        state = torch.stack(state)
        action = torch.stack(action)
        next_state = torch.stack(next_state)
        reward = torch.tensor(reward)
        terminated = torch.tensor(terminated).float().to(device)

        with torch.no_grad():
            target_q_values = reward + self.gamma * target_dqn(next_state).max(dim=1)[0] * (1 - terminated)

        current_q_values = policy_dqn(state).gather(1, action.unsqueeze(1)).squeeze()

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a DQN agent on Flappy Bird.")
    parser.add_argument("--train", action="store_true", help="Train the agent.")
    parser.add_argument('hyperparameters', help='')
    args = parser.parse_args()

    dql = Agent(args.hyperparameters)

    if args.train:
        dql.run(is_training=True, render=False)
    else:
        dql.run(is_training=False, render=True)

        