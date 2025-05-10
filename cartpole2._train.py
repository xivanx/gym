import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# 状态离散化函数（关键步骤）
def discretize_state(observation, bins=(10, 10, 10, 10)):
    """
    将连续状态的4个维度分别离散化为指定的bins
    CartPole观察空间: [cart位置, cart速度, pole角度, pole角速度]
    """
    discretized = []
    # 定义每个维度的离散化范围（根据实际环境调整）
    bounds = [
        (-4.4, 4.4),         # cart位置
        (-4.0, 4.0),         # cart速度
        (-0.25, 0.25),       # pole角度（约±14度）
        (-2.0, 2.0)          # pole角速度
    ]
    
    for i in range(len(observation)):
        # 将连续值映射到离散区间
        scale = (observation[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
        discretized_i = np.clip(int(scale * bins[i]), 0, bins[i]-1)
        discretized.append(discretized_i)
    return tuple(discretized)

class QLearningAgent:
    def __init__(
        self,
        env,
        learning_rate=0.1,
        initial_epsilon=1.0,
        epsilon_decay=0.9995,
        final_epsilon=0.01,
        discount_factor=0.95
    ):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        self.lr = learning_rate
        self.gamma = discount_factor
        
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.training_errors = []

    def get_action(self, obs):
        # 离散化观察值
        discrete_obs = discretize_state(obs)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # 探索
        else:
            return np.argmax(self.q_table[discrete_obs])  # 利用

    def update(self, obs, action, reward, next_obs, terminated):
        discrete_obs = discretize_state(obs)
        discrete_next_obs = discretize_state(next_obs)
        
        future_q = 0 if terminated else np.max(self.q_table[discrete_next_obs])
        td_error = reward + self.gamma * future_q - self.q_table[discrete_obs][action]
        
        self.q_table[discrete_obs][action] += self.lr * td_error
        self.training_errors.append(abs(td_error))
        
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

# 训练参数
n_episodes = 20_000
learning_rate = 0.1

# 初始化环境（训练时不渲染）
env_train = gym.make("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")

# 创建智能体
agent = QLearningAgent(
    env=env_train,
    learning_rate=learning_rate,
    initial_epsilon=1.0,
    epsilon_decay=0.9995,
    final_epsilon=0.01,
    discount_factor=0.95
)

# 训练循环
for episode in tqdm(range(n_episodes), desc="Training"):
    obs, _ = env_train.reset()
    done = False
    
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env_train.step(action)
        
        # 给奖励添加时间惩罚（防止智能体原地不动）
        reward = 1.0 if not terminated else -10.0
        
        agent.update(obs, action, reward, next_obs, terminated)
        done = terminated or truncated
        obs = next_obs
    
    agent.decay_epsilon()

env_train.close()

# 测试训练好的智能体（带渲染）
env_test = gym.make("CartPole-v1", render_mode="human")
total_rewards = []

for episode in range(10):  # 测试10次
    obs, _ = env_test.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(obs)  # 使用训练好的策略
        obs, reward, terminated, truncated, _ = env_test.step(action)
        total_reward += reward
        done = terminated or truncated
    
    total_rewards.append(total_reward)
    print(f"Test Episode {episode+1}, Total Reward: {total_reward}")

env_test.close()

print(f"\nAverage Test Reward: {np.mean(total_rewards):.2f}")

import pickle

# 保存训练好的 Q 表
with open('trained_q_table.pkl', 'wb') as f:
    # 将 defaultdict 转换为普通 dict 保存
    pickle.dump(dict(agent.q_table), f)

print("Q-table saved successfully.")