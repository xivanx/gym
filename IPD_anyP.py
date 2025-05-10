import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import os
import numpy as np

# 创建模型存储目录
MODEL_DIR = "saved_models1"
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

# 自定义随机初始状态包装器
class RandomResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 定义初始状态随机范围
        self.cart_pos_range = (-0.2, 0.2)    # 小车初始位置范围
        self.pole_angle_range = (-0.3, 0.3)  # 摆杆初始角度范围（弧度）
        self.velocity_range = (-0.05, 0.05)  # 初始速度范围

    def reset(self, **kwargs):
        # 调用原始reset方法
        obs, info = super().reset(**kwargs)
        
        # 生成随机初始状态
        qpos = self.unwrapped.data.qpos.flat.copy()
        qvel = self.unwrapped.data.qvel.flat.copy()
        
        # 设置小车位置
        qpos[0] = self.np_random.uniform(*self.cart_pos_range)
        # 设置摆杆角度
        qpos[1] = self.np_random.uniform(*self.pole_angle_range)
        qpos[2] = self.np_random.uniform(*self.pole_angle_range)
        # 设置速度
        qvel[:3] = self.np_random.uniform(*self.velocity_range, size=3)
        
        # 应用新状态
        self.unwrapped.set_state(qpos, qvel)
        return self.unwrapped._get_obs(), info

# 创建训练环境
def make_env():
    env = gym.make("InvertedDoublePendulum-v5", render_mode="rgb_array")
    env = RandomResetWrapper(env)
    return env

env = DummyVecEnv([make_env for _ in range(1)])

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

# 自动保存回调
class AutoSaveCallback(BaseCallback):
    def __init__(self, save_freq: int):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = os.path.join(MODEL_DIR, "ppo_double_pendulum_checkpoint")
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"{self.save_path}_{self.n_calls}")
        return True

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
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig(os.path.join(MODEL_DIR, "training_curve.png"))
plt.close()

# 模型加载演示函数
def load_and_play(model_path):
    print(f"\nLoading model from {model_path}...")
    
    loaded_model = PPO.load(
        model_path,
        env=make_env(),  # 使用相同的环境配置
        custom_objects={
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 64
        }
    )
    
    # 创建带随机初始化的渲染环境
    demo_env = RandomResetWrapper(gym.make("InvertedDoublePendulum-v5", render_mode="human"))
    obs, _ = demo_env.reset()
    
    for _ in range(1000):
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = demo_env.step(action)
        
        if terminated or truncated:
            obs, _ = demo_env.reset()
    
    demo_env.close()

# 演示最新模型
load_and_play("./saved_models1/ppo_double_pendulum_final.zip")