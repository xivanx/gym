import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 创建模型存储目录
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

class TrainingLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        return True


# 创建训练环境
env = gym.make("InvertedDoublePendulum-v5", render_mode="rgb_array")
env = DummyVecEnv([lambda: env])

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

# 训练配置
total_timesteps = 100000
logger = TrainingLogger()

# 带自动保存的回调函数
class AutoSaveCallback(BaseCallback):
    def __init__(self, save_freq: int):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = os.path.join(MODEL_DIR, "ppo_double_pendulum_checkpoint")
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"{self.save_path}_{self.n_calls}")
        return True

# 组合回调函数
callbacks = [logger, AutoSaveCallback(save_freq=10000)]

# 执行训练
model.learn(
    total_timesteps=total_timesteps,
    callback=callbacks,
    progress_bar=True,
)

# 最终模型保存
final_model_path = os.path.join(MODEL_DIR, "ppo_double_pendulum_final")
model.save(final_model_path)

# 训练结果可视化
plt.figure(figsize=(10, 6))
plt.plot(logger.episode_rewards)
plt.savefig(os.path.join(MODEL_DIR, "training_curve.png"))
plt.close()

# 模型加载演示函数
def load_and_play(model_path):
    print(f"\nLoading model from {model_path}...")
    
    # 加载时需要明确策略类型
    loaded_model = PPO.load(
        model_path,
        custom_objects={
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 64
        }
    )
    
    # 创建渲染环境
    demo_env = gym.make("InvertedDoublePendulum-v5", render_mode="human")
    obs, _ = demo_env.reset()
    
    for _ in range(1000):
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = demo_env.step(action)
        
        if terminated or truncated:
            obs, _ = demo_env.reset()
    
    demo_env.close()

# 演示最新模型
load_and_play("./saved_models/ppo_double_pendulum_final.zip")

# 可选：演示任意检查点
# load_and_play("saved_models/ppo_double_pendulum_checkpoint_50000")