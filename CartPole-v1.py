import gymnasium as gym

# 测试 CartPole 环境
env = gym.make("CartPole-v1", render_mode="human")
observation = env.reset()

for i in range(1000):
    action = env.action_space.sample()  # 随机动作
    observation, reward, terminated, truncated, info = env.step(action)
    print("this is number", i)
    
    if terminated or truncated:
        observation = env.reset()
env.close()