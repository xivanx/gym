import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from tqdm import tqdm

# 自定义回调函数用于收集训练数据
class TrainingLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # 累加奖励值
        self.current_episode_reward += self.locals['rewards'][0]
        
        # 检查episode是否结束
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        return True

# 创建训练环境
env = gym.make("InvertedDoublePendulum-v4", render_mode="rgb_array")
env = DummyVecEnv([lambda: env])  # 包装为向量环境

# 初始化PPO模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
)

# 训练参数设置
total_timesteps = 100000
progress_bar = tqdm(total=total_timesteps, desc="Training Progress")
logger = TrainingLogger()

# 自定义训练循环以支持进度条更新
model.learn(
    total_timesteps=total_timesteps,
    callback=[logger],
    progress_bar=True,
)

# 绘制训练结果
plt.figure(figsize=(10, 6))
plt.plot(logger.episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Performance")
plt.grid(True)
plt.show()

# 创建渲染环境并进行演示
print("\nStarting demonstration...")
demo_env = gym.make("InvertedDoublePendulum-v4", render_mode="human")
obs, _ = demo_env.reset()

for _ in range(50000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = demo_env.step(action)
    
    if terminated or truncated:
        obs, _ = demo_env.reset()

demo_env.close()