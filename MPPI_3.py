import gym
import torch
import random
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import DSAC_UAV_V16 as dsac
import methods
import pandas as pd
import time
import UAV_model

agent_model = UAV_model.UAV()
# set environment
env = gym.make('UAV-v0')

# DSAC
agent = dsac.DSAC(env)    # V16 DATA
agent.load()

#调用 DSAC 的 action
# _, action = agent.get_action(state)
# action = action.detach().cpu().numpy()

horizon = 18                                 # 预测域长度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
num_samples = 316                             #5 # 12  # sample number of DSAC policy and CEM policy
num_elites=6                                     #2
temperature = 0.5
momentum = 0.1
MPPI_iter = 6
STD = 1
class TOLD(nn.Module):
    def __init__(self):
        super().__init__()

        self._encoder = methods.enc(state_dim, env.reset() )

    @torch.no_grad()
    def plan(self, env, obs): 

        pi_actions = torch.empty(horizon, num_samples, action_dim)
        obs_pi = obs
        for i in range(num_samples):              # 依次取列数据，列数即数据数, 每一列都是相同的初值
            obs_pi = obs
            if i == 0:
               for t in range(horizon): 
                   _, pi_actions[t,i,:] = agent.get_action(obs_pi)    
                   obs_pi,r,_,_ = agent_model.step(obs_pi,pi_actions[t,i,:])
            else:
                for t in range(horizon): 
                    pi_actions[t,i,:] = agent.get_action_d(obs_pi)    
                    obs_pi,r,_,_ = agent_model.step(obs_pi,pi_actions[t,i,:])
               
        # print(pi_actions)
        mean = torch.zeros(horizon, action_dim)  
        std = STD * torch.ones(horizon, action_dim)  
        start_time = time.time()  
        for i in range(int(MPPI_iter)):  # J 
                
            if i == 0 :
                actions = pi_actions  
            else : 
                actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                          torch.randn(horizon, num_samples, action_dim), -4, 4)
            # print(actions)
            value = methods.estimate_value( obs, actions, horizon, 0, num_samples,action_dim)
            
            mppi_value = value 

            elite_idxs = torch.topk(value.squeeze(1), num_elites, dim=0).indices
            mppi_value_elite_idxs = torch.topk(mppi_value.squeeze(1), 1, dim=0).indices

            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]


            mppi_elite_value = mppi_value[mppi_value_elite_idxs].numpy().item()
            mppi_elite_actions = elite_actions[:, 0,:].numpy().squeeze()
            # print(i,'mppi_elite_value',mppi_elite_value,'mppi_elite_actions',mppi_elite_actions)
 

            # update MPPI parameter
            max_value = elite_value.max(0)[0]
            score = torch.exp(temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2,dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(horizon, 2)
            mean, std = momentum * mean + (1 - momentum) * _mean, _std
            # print('mean',mean)
        end_time = time.time()
        a = elite_actions[:, 0,:]
        #print('e_a',a)

        return a, mppi_elite_value, end_time - start_time

# # return 曲线
mean_m = 0
episode = 1000
Method = []
Return = []
Time = []
Std = []
Horizon = []
Iteration = []
Samples = []
for i in range (episode):
    state = env.reset()
    action, mppi_elite_value ,fortime= TOLD().plan(env,state)

    Method.append('MPPI_3')
    Return.append(mppi_elite_value)
    Samples.append(316)
    Time.append(fortime)
    Std.append(1)
    Horizon.append(18)
    Iteration.append(6)  
    print(i,mppi_elite_value,fortime)
    mean_m += mppi_elite_value
print('mean_m',mean_m/episode)

dataFrame_MPPI3_316 = pd.DataFrame(zip(Method, Return, Samples, Time, Std, Horizon, Iteration),\
                    columns = ['Method', 'Return', 'Samples', 'Time', 'Std', 'Horizon', 'Iteration'])
# dataFrame = pd.read_csv("/home/quyue/.vscode-server/.vscode/MD1/dataFrame.csv")
# dataFrame= dataFrame.append(dataFrame, ignore_index=True)
print(dataFrame_MPPI3_316)
dataFrame_MPPI3_316.to_csv("/home/quyue/.vscode-server/.vscode/MD1/dataFrame_MPPI3_316chong.csv",index=False,sep=',')
# return 曲线


# s
# A = np.array(action)
# path = "/home/quyue/.vscode-server/.vscode/MD1/trajectory/"
# path_A = os.path.join('%sM2_A%s' % (path,1))
# np.savetxt(path_A,A)
# print(mppi_elite_value)
