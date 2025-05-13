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
import DSAC_UAV_V20 as dsacg
import methods
import time
import UAV_model
import pandas as pd

agent_model = UAV_model.UAV()

# set environment
env = gym.make('UAV-v0')

# DSAC
agent = dsac.DSAC(env)    # V16 DATA
agent.load()

agentg = dsacg.DSAC(env)    # V15 DATA (better)
agentg.load()

horizon = 18                               # 预测域长度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
num_samples = 3162     #316  #32 #100   #   316 #32 #10 # 5 #12    #5 # 12  # sample number of DSAC policy and CEM policy
num_elites=64                               #6#4          #2
temperature = 1
momentum = 0.1
MPPI_iter = 6
STD = 0.5

class TOLD(nn.Module):
    def __init__(self):
        super().__init__()

        self._encoder = methods.enc(state_dim, env.reset() )

    @torch.no_grad()
    def plan(self, env,obs):
        PV = []  # DSAC policy value
        MV = []  # MPPI policy value
        PA = []  # DSAC policy
        MA = []  # MPPI policy

        # obs = env.reset(return_info=True)
        # obs = torch.tensor(obs).unsqueeze(0)  # transform state into tensor form

        num_pi_trajs = int(num_samples)
        pi_actions = torch.empty(horizon, num_pi_trajs, action_dim)

        # z = obs.repeat(num_pi_trajs, 1)        # 这个z是为了求出 pi_action 序列的
        # zz = obs.repeat(num_pi_trajs, 1)
        # calculate DSAC policy sequence based on model
        
        for i in range(num_pi_trajs ):              # 依次取列数据，列数即数据数,每一列都是相同的初值
            obs_pi = obs
            if i == 0:
               for t in range(horizon): 
                   _, pi_actions[t,i,:] = agent.get_action(obs_pi)    
                   obs_pi,r,_,_ = agent_model.step(obs_pi,pi_actions[t,i,:])
            else:
                for t in range(horizon): 
                    pi_actions[t,i,:] = agent.get_action_d(obs_pi)    
                    obs_pi,r,_,_ = agent_model.step(obs_pi,pi_actions[t,i,:])
        # for t in range(horizon):
        #     _, pi_actions[t] = agent.get_action(obs_pi)
        #     #print('pi_actions[t]',pi_actions[t][1])
        #     # obs,r,_,_ = env.step(pi_actions[t,0,:])
        #     obs_pi,r,_,_ = agent_model.step(obs_pi,pi_actions[t,0,:])
           
        # print('pi',pi_actions,pi_actions.shape)
        # obs = env.reset(return_info = False)
        # obs = torch.tensor(obs).unsqueeze(0)
        # z = obs.repeat(num_pi_trajs + num_samples, 1)       # estimate value function

        # mean = torch.zeros(horizon, action_dim)          # warm star
        mean = pi_actions[:,0,:]
        std = torch.ones(horizon, action_dim)
        start_time = time.time()
        for i in range(int(MPPI_iter)):  # J
            cem_actions = torch.clamp(mean.unsqueeze(1) + STD * std.unsqueeze(1) * \
                          torch.randn(horizon, num_samples, action_dim), -4, 4)

            # print('cem',cem_actions)
            # print('cem_1',cem_actions[0,0,:],cem_actions[0,0,:].shape)

            # MPPI sample number is twice as many as DSAC policy, which provide large optimization space
            if num_pi_trajs > 0:
                actions = torch.cat([pi_actions, cem_actions], dim=1)
                #print('a',actions)
            else:
                actions = cem_actions
            value = methods.estimate_value( obs, actions, horizon, num_pi_trajs, num_samples,action_dim)
            pi_value = value[0:num_pi_trajs]
            mppi_value = value[num_pi_trajs:]

            #print(i,'value',value,value.shape,type(value))
            # print(i,'pi_value',pi_value)
            # print(i,'mppi_value',mppi_value)

            elite_idxs = torch.topk(value.squeeze(1), num_elites, dim=0).indices
            pi_value_elite_idxs = torch.topk(pi_value.squeeze(1), 1, dim=0).indices
            mppi_value_elite_idxs = torch.topk(mppi_value.squeeze(1), 1, dim=0).indices

            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
            # print('elite_value',elite_value.shape,type(elite_value),elite_value)
            # print('elite_actions',elite_actions.shape,type(elite_actions),elite_actions)

            pi_elite_value, pi_elite_actions = pi_value[pi_value_elite_idxs].numpy().item(), \
                                               pi_actions[:, pi_value_elite_idxs,:].numpy().squeeze()

            # print(i,'pi_elite_value',pi_elite_value)
            # print('pi_elite_actions',pi_elite_actions.shape,type(pi_elite_actions),pi_elite_actions)

            mppi_elite_value = mppi_value[mppi_value_elite_idxs].numpy().item()
            mppi_elite_actions = elite_actions[:, 0,:].numpy().squeeze()
            # print(i,'mppi_elite_value',mppi_elite_value)
            # print('mppi_elite_actions',mppi_elite_actions.shape,type(mppi_elite_actions),mppi_elite_actions)

            MA.append(mppi_elite_actions)
            PV.append(pi_elite_value)
            PA.append(pi_elite_actions)
            MV.append(mppi_elite_value)

            # update MPPI parameter
            max_value = elite_value.max(0)[0]
            score = torch.exp(temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2,dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(horizon, 2)
            mean, std = momentum * mean + (1 - momentum) * _mean, _std
            # print('std',_std)
        end_time = time.time()

        a = elite_actions[:, 0,:]
        
        # print(end_time - start_time)
        #print('e_a',a)

        return a , pi_elite_value, mppi_elite_value, end_time - start_time


# return 曲线
mean_pi = 0
mean_md = 0
episode = 500
Method = []
Return = []
Time = []
Std = []
Horizon = []
Iteration = []
Samples = []
tem = []
for i in range (episode):
    state = env.reset()
    action, pi_elite_value, mppi_elite_value ,fortime= TOLD().plan(env,state)

    Method.append('MPPI_DSAC_2')
    Return.append(mppi_elite_value)
    Samples.append(3162)
    Time.append(fortime)
    Std.append(0.5)
    Horizon.append(18)
    Iteration.append(6)  
    tem.append(1) 
    
    print(i,'pi_elite_value',pi_elite_value,'mppi_elite_value',mppi_elite_value,'time',fortime)
    mean_md += mppi_elite_value
    mean_pi += pi_elite_value

print('mean_md',mean_md/episode)
print('mean_pi',mean_pi/episode)

dataFrame_MD2 = pd.DataFrame(zip(Method, Return, Samples, Time, Std, Horizon, Iteration,tem),\
                    columns = ['Method', 'Return', 'Samples', 'Time', 'Std', 'Horizon', 'Iteration','tem'])
# dataFrame = pd.read_csv("/home/quyue/.vscode-server/.vscode/MD1/dataFrame.csv")
# dataFrame = dataFrame.append(dataFrame1, ignore_index=True)
print(dataFrame_MD2 )
dataFrame_MD2 .to_csv("/root/quyue/data/dataFrame_MD2_num3162.csv",index=False,sep=',')
# return 曲线
5
# A = []
# score = 0
# state = env.reset(return_info = True)   #  固定初值return_info = True
# action, pi_elite_value, mppi_elite_value, Time = TOLD().plan(env,state)
# A = np.array(action)
# path = "/home/quyue/.vscode-server/.vscode/MD1/trajectory/"
# path_A = os.path.join('%sMD_2_A%s' % (path,1))
# np.savetxt(path_A,A)
# print(mppi_elite_value)

# 删除行
# dataFrame2 = pd.read_csv("/home/quyue/.vscode-server/.vscode/MD1/dataFrame2.csv")

# dataFrame2.drop(dataFrame1[dataFrame1['Samples']==32].index,inplace=True)
# print(dataFrame1)
# dataFrame2 = dataFrame2.replace(312, 316)
# dataFrame1.to_csv("/home/quyue/.vscode-server/.vscode/MD1/dataFrame2.csv",index=False,sep=',')


# mean_m = 0
# episode = 200
# for i in range (episode):
#     state = env.reset()
#     action, pi_elite_value, mppi_elite_value, MV, MA = TOLD().plan(env,state)
#     print(i,'pi_elite_value',pi_elite_value,'mppi_elite_value',mppi_elite_value)
#     mean_pi += pi_elite_value
#     mean_m += mppi_elite_value
# print('mean_pi',mean_pi/episode)
# print('mean_m',mean_m/episode)
'''
for episode in range(1):
            R = []
            S=[]
            score = 0
            state = env.reset(return_info = True)   #  固定初值return_info = True
            #print(state)
            for i in range(30):
                
                # _, action = agent.get_action(state)
                action, PV, PA, MV, MA = TOLD().plan(env,state)
                action = action.detach().cpu().numpy()
                # print(action)

                # M_action = MPPI_action[i,:]

                next_state, reward, done, _ = env.step(action)
                r_I,STATE = env.render()
                
                R.append(action)
                S.append(STATE)
                
                state = next_state
                 #  for plot mean value in each average_interval
                score += reward    #  for culculate return  in each episode/home/quyue/.vscode-server/.vscode/DATA
                if done:
                    break
R = np.array(R)
np.savetxt('/home/quyue/.vscode-server/.vscode/MD1/Action.txt',R)
print(score)
'''


# def pltbar():  # plot bar picture and line picture within one CEM iterate include 'MPPI_iter' steps
#     for i in range(1):
#         state = env.reset()
#         action, PV, PA, MV, MA = TOLD().plan(env)
#         # interact with environment
#         state, reward, done, _ = env.step(np.float32(action))

#     plt.figure()
#     x = list(range(len(PV)))
#     total_width, n = 0.8, 2
#     width = total_width / n
#     plt.bar(x, PV, width=width, label='PV', fc='deeppink', alpha=0.3)
#     for i in range(len(x)):
#         x[i] = x[i] + width
#     plt.bar(x, MV, width=width, label='MV', fc='cornflowerblue', alpha=0.3)
#     plt.plot(PA, color='deeppink', label='P_A', marker='o', markersize='6')
#     plt.plot(MA, color='cornflowerblue', label='M_A', marker='o', markersize='6')
#     plt.legend(loc="upper right")
#     plt.grid(True)
#     plt.show()


# def pltscore_1():  # plot n iterate return using DSAC and TDMPC and model estimate return 单步奖赏折线图,单步平均奖赏
#     score, score_, score_DSAC = 0, 0, 0
#     SCORE,SCORE_DSAC = 0, 0
#     R_TD,R_,R_DSAC = [],[],[]
#     mean_interval = 10
#     n = 5
#     for i in range(n):
#         state = env.reset()
#         action, PV, PA, MV, MA = TOLD().plan(env)
#         _, action_DSAC = agent.get_action(state)

#         next_state, reward, done, _ = env.step(np.float32(action))

#         state_tensor = torch.tensor(state).unsqueeze(0)
#         action_tensor = action.unsqueeze(0)
#         _, reward_ = agent2.get(state_tensor, action_tensor)
#         reward_ = reward_.detach().numpy()[0].item()

#         score_ += reward_  # for plot model estimate reward
#         score += reward  # for ture return
#         SCORE += reward  # for mean value

#         print(i,state,action,action_DSAC,)

#         if i % mean_interval == 0:
#             R_TD.append(score / mean_interval)
#             R_.append(score_ / mean_interval)
#             score, score_ = 0, 0

#     for i in range(n):
#         state = env.reset()

#         _, action_DSAC = agent.get_action(state)

#         next_state_DSAC, reward_DSAC, _, _ = env.step(np.float32(action_DSAC))

#         score_DSAC += reward_DSAC
#         SCORE_DSAC += reward_DSAC
#         print(i,state,action_DSAC)

#         if i % mean_interval == 0:
#             R_DSAC.append(score_DSAC / mean_interval)
#             score_DSAC = 0
#     print('TD',SCORE/n,'DSAC',SCORE_DSAC/n)

#     # plt.figure()
#     # plt.plot(R_TD, color='deeppink', label='R_TD')
#     # #plt.plot(R_, color='orange', label='R_')
#     # plt.plot(R_DSAC, color='cornflowerblue', label='R_DSAC')
#     # plt.ylabel('Return')
#     # plt.xlabel("Episode")
#     # plt.grid(True)
#     # plt.legend()
#     # plt.show()

# def pltscore_n():  # plot n iterate return using DSAC and TDMPC and model estimate return 多步累积奖赏折线图

#     R_TD,R_,R_DSAC = [],[],[]
#     mean_interval = 10
#     n = 300
#     Episode = 100

#     for episode in range(Episode) :
#         score, score_ = 0, 0
#         state= env.reset()
#         for i in range(n):
#             action, PV, PA, MV, MA = TOLD().plan(state)

#             next_state, reward, done, _ = env.step(np.float32(action))
#             if i % 1 == 0: env.render()

#             state_tensor = torch.tensor(state).unsqueeze(0)
#             action_tensor = action.unsqueeze(0)
#             _, reward_ = agent2.get(state_tensor, action_tensor)
#             reward_ = reward_.detach().numpy()[0].item()

#             state = next_state

#             score_ += reward_  # for plot model estimate reward
#             score += reward  # for ture reward

#             if done:
#                 break
#         print('a',action,episode,score,score_)
#         R_TD.append(score )
#         R_.append(score_ )

#     for episode in range(Episode):
#         score_DSAC = 0
#         state_DSAC = env.reset()
#         for i in range(n):
#             _, action_DSAC = agent.get_action(state_DSAC)

#             next_state_DSAC, reward_DSAC, _, _ = env.step(np.float32(action_DSAC))

#             state_DSAC = next_state_DSAC

#             score_DSAC += reward_DSAC
#             if done:
#                 break
#         print(episode)

#         R_DSAC.append(score_DSAC / mean_interval)

#     plt.figure()
#     plt.plot(R_TD, color='deeppink', label='R_TD')
#     plt.plot(R_, color='orange', label='R_')
#     plt.plot(R_DSAC, color='cornflowerblue', label='R_DSAC')
#     plt.ylabel('Return')
#     plt.xlabel("Episode")
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# pltbar()
