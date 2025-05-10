import time
import gymnasium as gym
from stable_baselines3 import PPO

def play_cartpole(model_path, episodes=3, max_steps=500, render_fps=50):
    """
    加载训练好的PPO模型进行实时演示
    参数：
        model_path: 模型文件路径
        episodes: 演示回合数
        max_steps: 每回合最大步数
        render_fps: 渲染帧率（每秒帧数）
    """
    try:
        # 加载训练好的模型
        model = PPO.load("./ppo_cartpole_tqdm.zip")
        print(f"成功加载模型: {"./ppo_cartpole_tqdm.zip"}")

        # 创建渲染环境
        env = gym.make('CartPole-v1', render_mode='human')
        
        for episode in range(1, episodes+1):
            obs, _ = env.reset()
            total_reward = 0
            step = 0
            
            print(f"\n=== 第 {episode} 回合 ===")
            for step in range(1, max_steps+1):
                # 使用模型预测动作
                action, _ = model.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                # 实时渲染（自动刷新）
                env.render()
                
                # 控制渲染速度
                time.sleep(1/render_fps)
                
                # 显示实时状态（每10步更新）
                if step % 10 == 0:
                    print(f"步数: {step:3d} | "
                          f"位置: {obs[0]:+6.2f} | "
                          f"速度: {obs[1]:+6.2f} | "
                          f"角度: {obs[2]:+6.2f} | "
                          f"角速度: {obs[3]:+6.2f}", end='\r')

                # 检查终止条件
                if terminated or truncated:
                    break

            # 回合结束统计
            print(f"\n回合 {episode} 结果: "
                  f"总步数 = {step}, 总奖励 = {total_reward:.1f}")
            
        # 关闭环境
        env.close()
        print("\n演示结束！")

    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "ppo_cartpole_tqdm.zip"  # 模型文件路径
    EPISODES = 3                         # 演示回合数
    MAX_STEPS = 500                      # 每回合最大步数
    FPS = 50                             # 渲染帧率

    # 开始演示
    play_cartpole(
        model_path=MODEL_PATH,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        render_fps=FPS
    )