from env.autocar import *
from PG.algo import PG
from PG.model import Model
from PG.agent import Agent
from DQN.replay_memory import ReplayMemory
import torch

LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 50000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 500  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.95  # reward 的衰减因子，一般取 0.9 到 0.999 不等
run = True
clock = pygame.time.Clock()
images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
        (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
env = ComputerCar(1, 1, PATH)
obs_dim = env.obs_dim
act_dim = env.act_dim
model = Model(obs_dim=obs_dim, act_dim=act_dim)
algorithm = PG(model, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    )

def run_train_episode(agent, env,run):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        clock.tick(60)
        draw(WIN, images, env)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        if run != True:
            break
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)
        # print("action:",action)
        obs, reward, done = env.step(action)
        reward_list.append(reward)
        if done:
            break
    pygame.quit()
    return obs_list, action_list, reward_list

def calc_reward_to_goal(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i+1 + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)

scores = []
n_games = 1000
run = True
win_num = 0
max_score=-10000
save_path = './dqn_car_model_test.pth'
for i in range(n_games):
    score = 0
    idx = 0
    done = False
    obs = env.reset()
#     if win_num >10:
#         print("game will finish here because the paras are best!")
#         break
    '''
    observation
    1. beta: sideslip angle (the angle between heading angle of the car and the center line)
    2. deviation: deviation between car and center line
    3. direction: check the car's direction from the center line(1 for left and -1 right) 
    '''
    obs_list, action_list, reward_list = [], [], []
    while not done:
        clock.tick(60)
        draw(WIN, images, env)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        if run != True:
            break
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)
        # print("action:",action)
        obs, reward, done = env.step(action)
        reward_list.append(reward)
        score = sum(reward_list)

        # print(observation)
    model.train()
    batch_obs = np.array(obs_list)
    batch_action = np.array(action_list)
    batch_reward = calc_reward_to_goal(reward_list)
    if win_num > 0:
        model = torch.load(save_path)
        algorithm = PG(model, gamma=GAMMA, lr=LEARNING_RATE)
        agent = Agent(
            algorithm,
            )
    agent.learn(batch_obs, batch_action, batch_reward)
    idx += 1
    print('episodes: ' + str(i) + '------score: ' + str(score)+'------win num: '+str(win_num))
    scores.append(score)
    if score > max_score:
        algorithm.save(save_path)
        max_score = score
    if score > 100000:
        win_num += 1
    if i > 100:
        pass
    if run != True:
        break
pygame.quit()