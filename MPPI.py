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
# import DSAC_UAV_V16 as dsac
import methods1
import pandas as pd
import time

# set environment
env = gym.make('UAV-v0')

# DSAC
# agent = dsac.DSAC(env)    # V16 DATA
# agent.load()

#调用 DSAC 的 action
# _, action = agent.get_action(state)
# action = action.detach().cpu().numpy()

horizon = 18                                 # 预测域长度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
num_samples = 100                              #5 # 12  # sample number of DSAC policy and CEM policy
num_elites=6                                     #2
temperature = 0.5
momentum = 0.1
MPPI_iter = 6
STD = 1
class TOLD(nn.Module):
    def __init__(self):
        super().__init__()

        self._encoder = methods1.enc(state_dim, env.reset() )

    @torch.no_grad()
    def plan(self, env, obs):
      
        mean = torch.zeros(horizon, action_dim)  # warm star
        std = STD * torch.ones(horizon, action_dim)
        start_time = time.time()
        for i in range(int(MPPI_iter)):  # J
            cem_actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                          torch.randn(horizon, num_samples, action_dim), -4, 4)
            actions = cem_actions
            value = methods1.estimate_value( obs, actions, horizon, 0, num_samples,action_dim)
            
            mppi_value = value

            elite_idxs = torch.topk(value.squeeze(1), num_elites, dim=0).indices
            mppi_value_elite_idxs = torch.topk(mppi_value.squeeze(1), 1, dim=0).indices

            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]


            mppi_elite_value = mppi_value[mppi_value_elite_idxs].numpy().item()
            mppi_elite_actions = elite_actions[:, 0,:].numpy().squeeze()
            # print(i,'mppi_elite_value',mppi_elite_value)
 

            # update MPPI parameter
            max_value = elite_value.max(0)[0]
            score = torch.exp(temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2,dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(horizon, 2)
            mean, std = momentum * mean + (1 - momentum) * _mean, _std
            #print('mean',type(mean),mean)
        end_time = time.time()
        a = elite_actions[:, 0,:]
        #print('e_a',a)

        return a, mppi_elite_value, end_time - start_time

mean_m = 0
episode = 500
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

    Method.append('MPPI')
    Return.append(mppi_elite_value)
    Samples.append(100)
    Time.append(fortime)
    Std.append(1)
    Horizon.append(18)
    Iteration.append(6)
    print(i,mppi_elite_value,fortime)
    mean_m += mppi_elite_value
print('mean_m',mean_m/episode)

# dataFrame_M_I6 = pd.DataFrame(zip(Method, Return, Samples, Time, Std, Horizon, Iteration),\
#                     columns = ['Method', 'Return', 'Samples', 'Time', 'Std', 'Horizon', 'Iteration'])
# dataFrame = pd.read_csv("/home/quyue/.vscode-server/.vscode/MD1/dataFrame.csv")
# dataFrame= dataFrame.append(dataFrame, ignore_index=True)
print(dataFrame_M_I6)
# dataFrame_M_I6.to_csv("/home/quyue/.vscode-server/.vscode/MD1/dataFrame_MPPI_I6.csv",index=False,sep=',')


