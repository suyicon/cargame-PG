import gym
import numpy as np
from agent import Agent
from model import Model
from algo import PG
from my_wrapper import MyWrapper

LEARNING_RATE = 1e-3


# 训练一个episode
def run_train_episode(agent, env):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()[0]
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)#随机采样，因为模型还没学习
        action_list.append(action)
       #print("action:",action)
        obs, reward, done, _ ,info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()[0]
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, isOver, _,_ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def calc_reward_to_goal(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i+1 + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)


def main():
    env =  MyWrapper()
    # env = env.unwrapped # Cancel the minimum score limit
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    alg = PG(model, lr=LEARNING_RATE)
    agent = Agent(alg)

    # 加载模型并评估
    # if os.path.exists('./model.ckpt'):
    #     agent.restore('./model.ckpt')
    #     run_evaluate_episodes(agent, env, render=True)
    #     exit()

    for i in range(2000):
        obs_list, action_list, reward_list = run_train_episode(agent, env)
        if i % 10 == 0:
            print("Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_goal(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            # render=True 查看显示效果
            total_reward = run_evaluate_episodes(agent, env, render=False)
            print('Episode {}, Test reward: {}'.format(i , total_reward))

    # save the parameters to ./model.ckpt
    #agent.save('./model.ckpt')


if __name__ == '__main__':
    main()