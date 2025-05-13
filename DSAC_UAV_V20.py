import torch
import random
import torch.nn as nn
import collections
import numpy as np
import time
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt
from torch.distributions import Normal
import argparse
import os
# import UAV1
import gym
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# directory = 'UAV-v16_data'
directory = '/root/quyue/UAV-v15_data'

# tensorboard --logdir=D:\work\TD_MPC_DSAC\data.view4\c_8_0-5_a_5_0-5
writer = SummaryWriter(directory)

# torch.set_num_threads(1)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument("--env_name", default="UAV-v0")
parser.add_argument("--file_name", default=directory)
parser.add_argument('--tau',  default=0.001, type=float) # target smoothing coefficient
parser.add_argument('--alpha',  default=0.1, type=float)
parser.add_argument('--critic_lr', type=float, default=0.00001, help='critic learning rate')
parser.add_argument('--actor_lr', default=0.00002, type=float)
parser.add_argument('--alpha_lr', default=0.00001, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--buffer_maxlen', default=1000000, type=int)  # buffer max size
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--num_hidden_cell', type=int, default=256)
parser.add_argument('--action_high', dest='list', type=float, default=[2.], action="append")
parser.add_argument('--action_low', dest='list', type=float, default=[-2.], action="append")
parser.add_argument('--TD_bound', type=float, default=10)

args = parser.parse_args()




class ReplayBeffer():
    def __init__(self, buffer_maxlen):
        self.buffer = collections.deque(maxlen=buffer_maxlen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience
            # state, action, reward, next_state, done

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)
        state = np.array(state_list)
        action = np.array(action_list)
        reward = np.array(reward_list)
        next_state = np.array(next_state_list)
        done = np.array(done_list)

        return torch.FloatTensor(state).to(device), \
               torch.FloatTensor(action).to(device), \
               torch.FloatTensor(reward).unsqueeze(-1).to(device), \
               torch.FloatTensor(next_state).to(device), \
               torch.FloatTensor(done).unsqueeze(-1).to(device)

    def buffer_len(self):
        return len(self.buffer)

 
def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            weight_shape = list(m.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class SoftQNet(nn.Module):
      def __init__(self, state_dim, action_dim, log_std_min=-0.1, log_std_max=4):
          super(SoftQNet, self).__init__()
          self.linear1 = nn.Linear(state_dim + action_dim, args.num_hidden_cell, bias=True)
          self.linear2 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)
          self.linear3 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)
          self.linear_mean_4 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)
          self.linear_mean_5 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)
          self.linear_std_4 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)
          self.linear_std_5 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)

          self.mean_layer = nn.Linear(args.num_hidden_cell, 1, bias=True)
          self.log_std_layer = nn.Linear(args.num_hidden_cell, 1, bias=True)
          self.log_std_min = log_std_min
          self.log_std_max = log_std_max
          self.denominator = max(abs(self.log_std_min), self.log_std_max)
          init_weights(self)


      def forward(self, state, action):
          
          x = torch.cat([state, action], 1)
          x = self.linear1(x)
          x = F.gelu(x)
          x = self.linear2(x)
          x = F.gelu(x)
          x = self.linear3(x)

          x_mean = F.gelu(x)
          x_mean = self.linear_mean_4(x_mean)
          x_mean = F.gelu(x_mean)
          x_mean = self.linear_mean_5(x_mean)
          x_mean = F.gelu(x_mean)

          x_std = F.gelu(x)
          x_std = self.linear_std_4(x_std)
          x_std = F.gelu(x_std)
          x_std = self.linear_std_5(x_std)
          x_std = F.gelu(x_std)

          mean = self.mean_layer(x_mean)
          log_std = self.log_std_layer(x_std)
          log_std = torch.clamp_min(self.log_std_max * torch.tanh(log_std / self.denominator), 0) + \
                    torch.clamp_max(-self.log_std_min * torch.tanh(log_std / self.denominator), 0)
          

          return mean, log_std

      def evaluate(self, state, action, device=torch.device("cpu"), min=False):
          mean, log_std = self.forward(state, action)
          std = log_std.exp()
          normal = Normal(torch.zeros(mean.shape), torch.ones(std.shape))

          if min == False:
             z = normal.sample().to(device)
             z = torch.clamp(z, -2, 2)
          elif min == True:
             z = -torch.abs(normal.sample()).to(device)
          q_value = mean.to(device) + torch.mul(z, std.to(device))

          return mean, std, q_value


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-5, log_std_max=1, edge=3e-3):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # self.action_high = torch.tensor(args.action_high, dtype=torch.float32)
        # self.action_low = torch.tensor(args.action_low, dtype=torch.float32)
        self.action_high = 4 * torch.ones(1)
        self.action_low = -4 * torch.ones(1)   # change with environment
        self.action_range = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2
        self.denominator = max(abs(self.log_std_min), self.log_std_max)

        self.linear1 = nn.Linear(state_dim, args.num_hidden_cell, bias=True)
        self.linear2 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)
        self.linear3 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)
        self.linear_mean_4 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)
        self.linear_mean_5 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)
        self.linear_std_4 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)
        self.linear_std_5 = nn.Linear(args.num_hidden_cell, args.num_hidden_cell, bias=True)

        self.mean_layer = nn.Linear(args.num_hidden_cell, 4, bias=True)
        self.log_std_layer = nn.Linear(args.num_hidden_cell, 4, bias=True)
        init_weights(self)


    # new
    def forward(self, state):
        x = self.linear1(state)
        x = F.gelu(x)
        x = self.linear2(x)
        x = F.gelu(x)
        x = self.linear3(x)

        x_mean = F.gelu(x)
        x_mean = self.linear_mean_4(x_mean)
        x_mean = F.gelu(x_mean)
        x_mean = self.linear_mean_5(x_mean)
        x_mean = F.gelu(x_mean)

        x_std = F.gelu(x)
        x_std = self.linear_std_4(x_std)
        x_std = F.gelu(x_std)
        x_std = self.linear_std_5(x_std)
        x_std = F.gelu(x_std)

        mean = self.mean_layer(x_mean)
        log_std = self.log_std_layer(x_std)
        log_std = torch.clamp_min(self.log_std_max * torch.tanh(log_std / self.denominator), 0) + \
                  torch.clamp_max(-self.log_std_min * torch.tanh(log_std / self.denominator), 0)

        return mean, log_std

    def action(self, state, epsilon=1e-4):

        # state = torch.FloatTensor(state)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = (torch.mul(self.action_range.to(device), action) + self.action_bias.to(device)).detach().cpu().numpy()

        return action, mean


    # Use re-parameterization tick
    def evaluate(self, state, device=torch.device("cpu"), epsilon=1e-4):
        #state = torch.FloatTensor(state)
        mean, log_std = self.forward(state)
        mean = mean.to(device)

        # mean_nan = torch.any(torch.isnan(mean))
        # log_std_nan = torch.any(torch.isnan(log_std))

        # if mean_nan :
        #     print('mean is nan')
        #     #mean = torch.where(torch.isnan(mean), torch.full_like(mean, 0), mean)

        # if log_std_nan :
        #     print('log_std is nan')
        
        log_std = log_std.to(device)
        normal = Normal(torch.zeros(mean.shape), torch.ones(log_std.shape))
        z = normal.sample().to(device)
        std = log_std.exp()
        z = torch.clamp(z, -3, 3)
        action_0 = mean + torch.mul(z, std)
        action_1 = torch.tanh(action_0)
        action = torch.mul(self.action_range.to(device), action_1) + self.action_bias.to(device)
        log_prob = Normal(mean, std).log_prob(action_0) - torch.log(1. - action_1.pow(2) + epsilon)\
                   - torch.log(self.action_range.to(device))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, std.detach()

'''
env=gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
# q=SoftQNet(state_dim, action_dim)
state = env.reset()
p=PolicyNet(state_dim, action_dim)
a,b,k=p.action(state)
c,d,e= p.evaluate(state)
print(type(a),type(b),type(k),type(e))
# state=torch.FloatTensor(state)
# Q=q.forward(state, a)
'''

class DSAC:
    def __init__(self, env):
        self.env = env
        self.state_dim = 13
        self.action_dim = 4
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.action_high = 200 * np.ones(1) # * torch.ones(1)
        # self.action_low = -200 * np.ones(1) # * torch.ones(1)

        self.gamma = args.gamma
        self.tau = args.tau

        self.q1_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q1_net_target = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(device)   # self.action_dim=1
        self.policy_net_target = PolicyNet(self.state_dim, self.action_dim).to(device)

        # update the target network parameters
        for target_param, param in zip(self.q1_net_target.parameters(), self.q1_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.policy_net_target.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Initialize the optimizer
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=args.critic_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.actor_lr)

        # alpha tuning
        self.automatic_alpha_tuning = True   # True or False
        if self.automatic_alpha_tuning is True:
            self.target_entropy = -4 #-torch.prod(torch.Tensor(self.env.action_space.shape)).item() # H_0
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=args.alpha_lr)
            self.alpha = self.log_alpha.exp().to(device)
        else:
            self.alpha = args.alpha

        # initialize replay buffer
        self.buffer = ReplayBeffer(args.buffer_maxlen)

    def get_action(self, state):  # interact with the environment
       
        state = torch.FloatTensor(state).to(device)
        action, log_prob ,_= self.policy_net.evaluate(state)  # for train
        _, mean = self.policy_net.action(state)  # for test

        return action, mean

    def get_action_d(self, state):  # interact with the environment
        state = torch.FloatTensor(state).to(device)
        _, log_prob , action= self.policy_net.evaluate(state)  # for train

        return action
        
    def get_action_tensor(self, state):  # interact with the environment
        state = torch.FloatTensor(state)
        action, mean = self.policy_net.action(state)
        mean = mean.detach().cpu().numpy()
        action_range = [self.action_low, self.action_high]
        mean_in =mean * (action_range[1] - action_range[0]) / 2.0 + (
                 action_range[1] + action_range[0]) / 2.0
        mean_in = torch.FloatTensor(mean_in)

        return  mean_in.detach()

    def get_qloss(self,q, q_std, target_q, target_q_bound):
        loss = torch.mean(torch.pow(q - target_q, 2) / (2 * torch.pow(q_std.detach(), 2)) \
                          + torch.pow(q.detach() - target_q_bound, 2) / (2 * torch.pow(q_std, 2)) \
                          + torch.log(q_std))

        return loss
    
    def send_to_device(self,s, a, r, s_next,  done, device):
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_next = s_next.to(device)
        done = done.to(device)
        return s, a, r, s_next,  done


    def target_q(self, r, done, q, q_std, q_next, log_prob_a_next):
        target_q = r + (done) * args.gamma * (q_next - self.alpha.detach() * log_prob_a_next)
        difference = torch.clamp(target_q - q, - args.TD_bound, args.TD_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

    def update(self, batch_size):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        state, action, reward, next_state, done = self.send_to_device(state, action, reward, next_state, done, device)
        # print('model has been update')

        # compute for policy
        action1, log_prob, _ = self.policy_net.evaluate(state,device = device)
        q_1_policy,_,_ = self.q1_net.evaluate(state, action1, device = device)

        # compute for q
        new_action, log_prob_next, _ = self.policy_net_target.evaluate(next_state,device = device)  # compute for target_q_1, target_q_1_bound
        q_1, q_std_1, _ = self.q1_net.evaluate(state, action,device = device)
        _, _, q_next_target = self.q1_net_target.evaluate(next_state, new_action,device = device)
        target_q_1, target_q_1_bound = self.target_q(reward, done, q_1.detach(), q_std_1.detach(), q_next_target.detach(),log_prob_next.detach())

        # q loss
        q_loss_1 = self.get_qloss(q_1, q_std_1, target_q_1, target_q_1_bound)

        # policy loss
        policy_loss = (self.alpha.detach() * log_prob - q_1_policy).mean()

        # Update Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        # Update Soft q
        self.q1_optimizer.zero_grad()
        q_loss_1.backward()
        self.q1_optimizer.step()

        # alpha loss and Update alpha
        if self.automatic_alpha_tuning is True:
            alpha_loss = - (self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # update the target network parameters
        for target_param, param in zip(self.q1_net_target.parameters(), self.q1_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.policy_net_target.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        return (-(log_prob.detach().cpu().numpy().sum())/args.batch_size), \
               self.target_entropy,\
               (q_1.detach().cpu().numpy().sum())/args.batch_size,\
               (q_std_1.detach().cpu().numpy().sum())/args.batch_size


    def get_alpha(self):    # 也可以在 update 模块直接输出
        alpha = self.alpha.detach().cpu().numpy()   # action = torch.tanh()
        return alpha

    def get_q(self,state):    # 也可以在 update 模块直接输出
        state = torch.FloatTensor(state).to(device)
        action1, _, _ = self.policy_net.evaluate(state.to(device))   # on GPU
        action1 =  action1.to(device)
        # print(state.unsqueeze(0).shape, action1.unsqueeze(0).shape)
        # x = torch.cat([state.unsqueeze(0), action1.unsqueeze(0)], 1)
        
        q, _, _ = self.q1_net.evaluate(state.unsqueeze(0), action1.unsqueeze(0))
        return (q/5).detach().cpu().numpy().squeeze()

    def save(self):
        torch.save(self.policy_net.state_dict(), args.file_name + '_actor.pth')
        torch.save(self.q1_net.state_dict(), args.file_name + '_critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.policy_net.load_state_dict(torch.load(args.file_name + '_actor.pth', map_location='cuda:0'))
        self.q1_net.load_state_dict(torch.load(args.file_name + '_critic.pth', map_location='cuda:0'))
        # print("====================================")
        # print("model has been loaded...")
        # print("====================================")


def main(env,  agent, Episode, batch_size):
    if args.mode == 'train':
        mean_score = 0
        mean_average = 20
        sum_score = 0
     
        for episode in range(Episode):
            score = 0
            alpha = 0
            q = 0
            log_prob = 0
            q_std = 0
            state = env.reset()           
            start_time = time.time()
          
            for i in range(30):
                action, _ = agent.get_action(state)
                action = action.detach().numpy()

                action_nan = np.any(np.isnan(action))
                if action_nan :
                 print('action is nan')
                
                next_state, reward, done, r_I= env.step(action)

                next_state_nan = np.any(np.isnan(next_state))
                if next_state_nan :
                 print('next_state is nan')

                done_mask = 0.0 if done else 1.0

                agent.buffer.push((state, action, reward * 5, next_state, done_mask))

                state = next_state

                score += reward

                if agent.buffer.buffer_len() > 600:
                    log_prob, target_entropy, q, q_std = agent.update(batch_size)
                    alpha = agent.get_alpha()
                
                    if i % 5 == 0:
                        agent.save()

                if done:                    
                    break
            end_time = time.time()

            sum_score += score
            if episode % mean_average ==0 :
                mean_score = sum_score / mean_average
                sum_score = 0
                print('mean_score',mean_score)
           
            if episode % 50 ==0 :      # 固定初值
                state_test = env.reset(return_info = True)
                score_test = 0
                for j in range(300):
                    _, action_test = agent.get_action(state_test)
                    action_test = action_test.detach().cpu().numpy()
                    next_state_test, reward_test, done_test, _  = env.step(action_test)
                    state_test = next_state_test
                    score_test += reward_test
                    if done_test:            
                        break

            print("episode:{}, state:{},Return:{}, action:{}, buffer_len:{}".format(episode, state, score,  action, agent.buffer.buffer_len()))

            writer.add_scalar('Return in each episode(test)', score_test, episode)
            writer.add_scalar('Average Return within 20 episode(train)', mean_score, episode)
            writer.add_scalar('alpha', alpha, episode)
            writer.add_scalar('Q', q, episode)
            writer.add_scalar('log_prob', log_prob, episode)
            writer.add_scalar('q_std', q_std, episode)
            writer.add_scalar('each_episode_time', (end_time - start_time), episode)



    elif args.mode == 'test':
        agent.load()
        Return = []
        R = []
        #action_range = [env.action_space.low, env.action_space.high]
        score_all = 0  # evaluate tne mean of reward
        score1 = 0

        average_interval = 10    # compute average in the range of 'averange_interval'
        for episode in range(Episode):
            score = 0
            state = env.reset()
            for i in range(30):
                action, _ = agent.get_action(state)
                action = action.detach().numpy()
          

                next_state, reward, done, _ = env.step(state, action)
                #print(done)
                R.append(r_I)
                #next_state, reward, done, _ = env.step(np.float32(action))
                state = next_state
                score1 += reward   #  for plot mean value in each average_interval
                score += reward    #  for culculate return in each episode
                if done:
                    break
                #if i % 10 == 0: env.render()
            #print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent.buffer.buffer_len()))
            if episode % average_interval == 0 :
               Return.append(score1 / average_interval)
               score1 = 0

            score_all += score
        mean_reward = score_all / Episode  # evaluate the reward from episode 100 to 1000
        R = np.array(R)
        #print(R,R.shape,type(R))

        plt.figure(1)
        plt.plot(Return)
        plt.ylabel('Return')
        plt.xlabel("Episode")
        plt.grid(True)
        plt.show()

        #print('mean reward', mean_reward)

    #env.close()


if __name__ == '__main__':
    env = gym.make('UAV-v0')
 
    device = torch.device('cuda')

    Episode = 300000
    batch_size = 256

    agent = DSAC(env)
    # main(env,agent, Episode, batch_size)

