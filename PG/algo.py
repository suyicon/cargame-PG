import torch

class PG:
    def __init__(self,model,lr = None,gamma=0.99):
        self.model = model
        assert isinstance(lr,float)
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=self.lr)

    def predict(self,obs):
        prob = self.model(obs)
        return prob

    def learn(self,obs,action,reward):
        prob = self.model(obs)#prob(s,a*2)
        action_dim = prob.shape[-1]
        action = torch.squeeze(action,axis = -1)
        action_onehot = torch.eye(action_dim)[action]#[32,2]
        log_prob = torch.sum(torch.log(prob)*action_onehot,axis=1)#按列相加,交叉熵
        #log_prob = torch.nn.CrossEntropyLoss(prob,action_onehot)
        loss = torch.mean(-1*log_prob*reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    def save(self,path):
        torch.save(self.model, path)
