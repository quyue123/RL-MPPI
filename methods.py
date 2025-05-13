import gym
import torch
import random
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt
from torch.distributions import Normal
import argparse
import os
import DSAC_UAV_V16 as dsac
import UAV_model
import DSAC_UAV_V20 as dsacg

env = gym.make('UAV-v0')
agent_model = UAV_model.UAV()

agentg = dsacg.DSAC(env)    # V15 DATA (better)
agentg.load()

@torch.no_grad()
def estimate_value( obs,actions, horizon, num_pi_trajs, num_samples, action_dim):

        agent = dsac.DSAC(env)
        agent.load()

        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount,Rreward = 0, 1,0
        # print(actions)
        Value_Box = torch.empty(num_pi_trajs + num_samples,1)  # 储存 pi_action 和 cem_action 的动作的价值序列，是 tensor 格式的
        # zz = torch.tensor(obs).unsqueeze(0).repeat(num_pi_trajs + num_samples, 1)
        for i in range(num_pi_trajs + num_samples):              # 依次取列数据，列数即数据数
                # z_pi = env.reset(return_info=True)
                # print('rset_z',z_pi)
                Reward = 0
                obs1 = obs
                for t in range(horizon):                         #  预测域：取列向量
                        pi_action = actions[t,i,:]               # 三维矩阵，先行后列，维度
                        # print('pi_action',i,t,pi_action)
                        zz,reward,done,_ = agent_model.step(obs1,pi_action)    #  pi_reward : 单步horizon的奖赏
                        obs1 = zz
                        # print('reward',reward)
                        Reward += reward
                        if done:
                                break
                        # print('zz',zz)
                # print('value111',Reward)
                if not done:
                   Reward = Reward + agentg.get_q(zz)
                # print('zhen_pi',Reward)
                # print('zz',zz,'Q',agent.get_q(zz))
                Value_Box[i] =  Reward
        return Value_Box        
        # print('Value_Box',Value_Box)


        # for t in range(horizon):
        #         z, reward = agent2.get(z, actions[t])
        #         # compute pi_actions value
        #         z_pi = torch.tensor(env.reset(return_info=False)).unsqueeze(0).repeat(num_pi_trajs, 1)
        #
        #
        #         # obs = torch.tensor(obs).unsqueeze(0)
        #         # z = obs
        #         # zz, _, _, _ = env.step(pi_actions[t])
        #         # zz = torch.tensor(zz).unsqueeze(0).repeat(num_samples, 1)
        #         # print()
        #         #next_state, reward, done, _ = env.step(action_in)    #  不能同时求多条轨迹
        #         G += discount * reward
        #         discount *= discount
        # q = agent.get_q(z)
        # G += discount * q
        # G = torch.rand(10,1)
        
# @torch.no_grad()
# def estimate_value( z, actions,horizon):
#
#         agent = dsac.DSAC(env)
#         agent.load()
#
#         agent2 = model.QY()
#         agent2.load()
#
#         agent3 = sacqq.SAC(env)  # SAC
#         agent3.load()
#
#        # obs = env.reset(return_info=False)
#
#         """Estimate value of a trajectory starting at latent state z and executing given actions."""
#         G, discount = 0, 1
#         # zz = torch.tensor(obs).unsqueeze(0).repeat(num_pi_trajs + num_samples, 1)
#
#         for t in range(horizon):
#                 z, reward = agent2.get(z, actions[t])
#
#                 # obs = torch.tensor(obs).unsqueeze(0)
#                 # z = obs
#                 # zz, _, _, _ = env.step(pi_actions[t])
#                 # zz = torch.tensor(zz).unsqueeze(0).repeat(num_samples, 1)
#                 # print()
#                 #next_state, reward, done, _ = env.step(action_in)    #  不能同时求多条轨迹
#                 G += discount * reward
#                 discount *= discount
#         q = agent.get_q(z)
#         G += discount * q
#         return G
# @torch.no_grad()
# def enc(state_dim):
# 	"""Returns a TOLD encoder."""
# 	layers = [nn.Linear(state_dim, 256), nn.ELU(),
# 	 			  nn.Linear(256, state_dim)]
# 	return nn.Sequential(*layers)
def enc(state_dim,obs):
	"""Returns a TOLD encoder."""

	return obs
