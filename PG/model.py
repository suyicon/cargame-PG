import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, obs_dim,act_dim):
        super().__init__()
        hid_size = 256
        # 3层全连接网络
        self.fc1 = torch.nn.Linear(in_features=obs_dim,out_features=hid_size)
        self.fc2 = torch.nn.Linear(in_features=hid_size,out_features=act_dim)

    def forward(self, obs):
        h1 = self.fc1(obs)
        h1 = F.tanh(h1)
        h2 = self.fc2(h1)
        prob = F.softmax(h2)
        #print(prob)
        return prob