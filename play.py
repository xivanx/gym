import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle

# 必须与训练代码完全一致的离散化函数
def discretize_state(observation, bins=(10, 10, 10, 10)):
    """与训练时相同的状态离散化逻辑"""
    discretized = []
    bounds = [
        (-2.4, 2.4),         # cart位置
        (-3.0, 3.0),         # cart速度
        (-0.25, 0.25),       # pole角度（约±14度）
        (-2.0, 2.0)          # pole角速度
    ]
    
    for i in range(len(observation)):
        scale = (observation[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
        discretized_i = np.clip(int(scale * bins[i]), 0, bins[i]-1)
        discretized.append(discretized_i)
    return tuple(discretized)

# 加载训练好的 Q 表
with open('trained_q_table.pkl', 'rb') as f:
    q_table_dict = pickle.load(f)

# 创建环境（启用渲染）
env = gym.make("CartPole-v1", render_mode="human")

# 将普通字典转换为 defaultdict
q_table = defaultdict(lambda: np.zeros(env.action_space.n))  
q_table.update(q_table_dict)

# 运行测试
total_rewards = []
for episode in range(10):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 离散化当前状态
        discrete_state = discretize_state(obs)
        
        # 选择 Q 值最大的动作
        action = np.argmax(q_table[discrete_state])
        
        # 执行动作
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    total_rewards.append(total_reward)
    print(f"Test Episode {episode+1}, Total Reward: {total_reward}")

env.close()
print(f"\nAverage Test Reward: {np.mean(total_rewards):.2f}")