import gymnasium as gym
from stable_baselines3 import PPO
import os

class RandomStartWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.model = env.unwrapped.model  # 访问MuJoCo模型
        self.data = env.unwrapped.data    # 访问MuJoCo数据
    
    def reset(self, **kwargs):
        # 生成随机初始状态（qpos和qvel）
        initial_qpos = np.array([
            self.np_random.uniform(low=-0.5, high=0.5),   # 小车位置
            self.np_random.uniform(low=-np.pi, high=np.pi),  # 摆杆1角度
            self.np_random.uniform(low=-np.pi, high=np.pi)   # 摆杆2角度
        ])
        initial_qvel = self.np_random.uniform(low=-1, high=1, size=self.model.nv)
        
        # 设置模型状态
        self.unwrapped.set_state(initial_qpos, initial_qvel)
        return self.unwrapped._get_obs(), {}

def play_saved_model(model_path, render_mode="human", max_steps=1000):
    """
    加载并运行已保存的模型进行演示
    :param model_path: 模型文件路径
    :param render_mode: 渲染模式（默认human可见）
    :param max_steps: 最大运行步数（默认1000步）
    """
    # 加载模型
    model = PPO.load(model_path)

    # 创建带渲染的环境
    env = gym.make("InvertedDoublePendulum-v5", render_mode=render_mode)
    obs, _ = env.reset()
    
    try:
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            
            # 重置环境如果episode结束
            if terminated or truncated:
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\n演示已手动终止")
    finally:
        env.close()
        print("环境已关闭")

if __name__ == "__main__":
    # 配置参数（根据需要修改）
    MODEL_PATH = "saved_models1/ppo_double_pendulum_final"  # 模型路径
    RENDER_MODE = "human"  # 可选：rgb_array（不显示窗口）
    DEMO_STEPS = 20000      # 演示步数
    
    # 运行演示
    play_saved_model(
        model_path=MODEL_PATH,
        render_mode=RENDER_MODE,
        max_steps=DEMO_STEPS
    )  