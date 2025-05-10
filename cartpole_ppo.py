import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # 自动选择合适环境的tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from IPython.display import HTML
from matplotlib import animation

# 创建环境
env = gym.make('CartPole-v1')
env = Monitor(env)

# 自定义回调函数集成tqdm进度条
class TQDMCallback(BaseCallback):
    def __init__(self, total_timesteps, check_freq=1000, verbose=0):
        super(TQDMCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.total_timesteps = total_timesteps
        self.progress_bar = None
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_training_start(self):
        # 初始化进度条
        self.progress_bar = tqdm(total=self.total_timesteps, 
                                desc="Training Progress",
                                dynamic_ncols=True)
        
    def _on_step(self) -> bool:
        # 每隔check_freq步更新进度条
        if self.n_calls % self.check_freq == 0:
            # 获取最近的训练数据
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]
                self.episode_rewards.extend(rewards)
                self.episode_lengths.extend(lengths)
                
                # 更新进度条描述
                avg_reward = np.mean(rewards)
                avg_length = np.mean(lengths)
                self.progress_bar.set_postfix(
                    avg_reward=f"{avg_reward:.2f}",
                    avg_length=f"{avg_length:.2f}"
                )
            
            # 更新进度条（按实际步数）
            current_step = self.model.num_timesteps
            self.progress_bar.n = current_step
            self.progress_bar.refresh()
            
        return True
    
    def _on_training_end(self):
        # 确保进度条完成并关闭
        self.progress_bar.n = self.total_timesteps
        self.progress_bar.close()

# 训练参数
total_timesteps = 50000
check_freq = 1000  # 进度更新频率

# 初始化模型和回调
model = PPO("MlpPolicy", env, verbose=0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2)

callback = TQDMCallback(total_timesteps=total_timesteps, check_freq=check_freq)

# 开始训练
model.learn(total_timesteps=total_timesteps, callback=callback)

# 保存模型
model.save("ppo_cartpole_tqdm")

# 可视化训练结果（与之前相同）
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(np.convolve(callback.episode_rewards, np.ones(10)/10, mode='valid'))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards (Moving Average)')

plt.subplot(1, 2, 2)
plt.plot(np.convolve(callback.episode_lengths, np.ones(10)/10, mode='valid'))
plt.xlabel('Episode')
plt.ylabel('Length')
plt.title('Episode Lengths (Moving Average)')

plt.tight_layout()
plt.show()

# 运行动画演示
def save_frames_as_gif(frames, path='./cartpole_ppo_tqdm.gif'):
    plt.figure(figsize=(6, 6))
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path, writer='pillow', fps=30)

env = gym.make('CartPole-v1', render_mode='rgb_array')
obs, _ = env.reset()
frames = []

for _ in range(200):
    action, _ = model.predict(obs)
    obs, _, terminated, truncated, _ = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        break

env.close()
save_frames_as_gif(frames)

# 在notebook中显示结果
HTML('<img src="cartpole_ppo_tqdm.gif">')

# 新增实时play环节
def play_cartpole(model, episodes=3, max_steps=500):
    for episode in range(1, episodes+1):
        # 创建渲染环境
        env = gym.make('CartPole-v1', render_mode='human')
        obs, _ = env.reset()
        total_reward = 0
        frames = []
        
        print(f"\n=== Episode {episode} ===")
        for step in range(1, max_steps+1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # 实时显示当前状态
            env.render()  # 对于某些版本可能需要显式调用
            
            # 打印实时信息（可选）
            print(f"Step {step}: Position={obs[0]:.2f}, "
                  f"Velocity={obs[1]:.2f}, "
                  f"Angle={obs[2]:.2f}, "
                  f"Angular Velocity={obs[3]:.2f}", 
                  end='\r', flush=True)
            
            if terminated or truncated:
                print(f"\nEpisode {episode} 结束！总奖励: {total_reward} 步数: {step}")
                break
        else:
            print(f"\nEpisode {episode} 达到最大步数！总奖励: {total_reward}")
        
        env.close()

# 执行play（训练完成后调用）
print("\n开始实时演示...")
play_cartpole(model, episodes=3)

# 可选：保留之前的GIF生成代码