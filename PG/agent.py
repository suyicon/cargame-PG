import torch
import numpy as np

class Agent:
    def __init__(self,algo):
        self.algo = algo

    def sample(self,obs):
        #print("obs:",obs)
        obs = torch.FloatTensor(obs)
        prob = self.algo.predict(obs)
        prob = prob.detach().numpy()
        action = np.random.choice(len(prob),size=1,p=prob)[0]#[action]->action
        return action
    def predict(self,obs):
        obs = torch.FloatTensor(obs)
        prob = self.algo.predict(obs)
        action = int(prob.argmax())
        return action
    def learn(self,obs,action,reward):
        action = np.expand_dims(action,axis=-1)
        reward = np.expand_dims(reward,axis=-1)

        obs = torch.FloatTensor(obs)
        action = torch.IntTensor(action)
        reward = torch.FloatTensor(reward)#这里的reward是Gt

        loss = self.algo.learn(obs,action,reward)
        return float(loss)