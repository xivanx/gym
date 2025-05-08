import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

# 状态离散化函数（向量化版本）
def discretize_state_vector(observations, bins=(6, 6, 12, 12)):
    bounds = np.array([
        [-2.4, 2.4], [-3.0, 3.0], [-0.25, 0.25], [-2.0, 2.0]
    ])
    scaled = (observations - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    discretized = np.clip((scaled * bins).astype(int), 0, np.array(bins) - 1)
    return [tuple(row) for row in discretized]

class VectorQLearningAgent:
    def __init__(
        self,
        num_envs=4,
        learning_rate=0.15,
        initial_epsilon=1.0,
        epsilon_decay=0.9998,
        final_epsilon=0.02,
        discount_factor=0.97,
        bins=(6, 6, 12, 12)
    ):
        self.num_envs = num_envs
        self.env = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(num_envs)])
        self.q_table = defaultdict(lambda: np.zeros(self.env.single_action_space.n))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.bins = bins
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_errors = []

    def get_actions(self, observations):
        discrete_states = discretize_state_vector(observations, self.bins)
        actions = []
        for state in discrete_states:
            if np.random.random() < self.epsilon:
                actions.append(self.env.single_action_space.sample())
            else:
                actions.append(np.argmax(self.q_table[state]))
        return np.array(actions)

    def batch_update(self, batch):
        for transition in batch:
            obs, action, reward, next_obs, terminated = transition
            discrete_obs = discretize_state_vector([obs], self.bins)[0]
            discrete_next_obs = discretize_state_vector([next_obs], self.bins)[0]
            future_q = 0 if terminated else np.max(self.q_table[discrete_next_obs])
            td_error = reward + self.gamma * future_q - self.q_table[discrete_obs][action]
            self.q_table[discrete_obs][action] += self.lr * td_error
            self.training_errors.append(abs(td_error))

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, total_timesteps=100_000):
        batch_size = self.num_envs * 4
        obs, _ = self.env.reset()
        progress = tqdm(total=total_timesteps, desc="Training")
        
        batch = []
        timestep = 0
        
        while timestep < total_timesteps:
            actions = self.get_actions(obs)
            next_obs, rewards, terminateds, truncateds, _ = self.env.step(actions)
            
            for i in range(self.num_envs):
                done = terminateds[i] or truncateds[i]
                batch.append((obs[i], actions[i], rewards[i] if not done else -10.0, next_obs[i], done))
                if done:
                    # ✅ 修复重置逻辑
                    next_obs[i] = self.env.reset(indices=[i])[0][0]
            
            if len(batch) >= batch_size:
                self.batch_update(batch)
                batch = []
                self.decay_epsilon()
                timestep += batch_size
                progress.update(batch_size)
        
        self.env.close()
        progress.close()

    def demo(self, episodes=5):
        env = gym.make("CartPole-v1", render_mode="human")
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = self.get_actions(np.array([obs]))[0]
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        env.close()

# 训练与演示
agent = VectorQLearningAgent(num_envs=4)
agent.train(total_timesteps=50_000)
agent.demo()