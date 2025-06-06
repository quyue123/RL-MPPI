from os import path
#from subprocess import HIGH_PRIORITY_CLASS
from typing import Optional
import math
import numpy as np
import gym
from gym import spaces, core
# gym.make('UAV-v0')


class UAV(gym.Env):
    def __init__(self, g=10.0):
        super(UAV).__init__()
        # dyna para
        self.g = g
        self.Jx = 1
        self.Jy = 1
        self.Jz = 1
        self.m = 1
        self.c = 0.01
        self.l = 0.4
        self.J_B = np.diag(np.array([self.Jx,self.Jy,self.Jz]))
        self.g_I = np.array([0, 0, -self.g])
        self.dt = 0.1

        # reward para
        self.wr = 0.02
        self.wv = 0.001
        self.wq = 0.002
        self.ww = 0.001
        self.wthrust = 0.0001

        self.action_high = np.array([4,4,4,4], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.action_high, high=self.action_high, shape=(4,), dtype=np.float32)
    

        self.high = np.array([1000, 1000, 1000, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10], dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.high, high=self.high, dtype=np.float32)


    def step(self, u):

        rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz = self.state
        u = np.clip(u, -self.action_high, self.action_high)
        f1, f2, f3, f4 = u
        self.r_I = np.array([rx, ry, rz])
        self.v_I = np.array([vx, vy, vz])
        self.q = np.array([q0, q1, q2, q3])
        self.w_B = np.array([wx, wy, wz])
        self.T_B = np.array([f1, f2, f3, f4])
        thrust = (self.T_B[0] + self.T_B[1] + self.T_B[2] + self.T_B[3])
        self.thrust_B = np.array([0, 0, thrust])

        Mx = -self.T_B[1] * self.l / 2 + self.T_B[3] * self.l / 2
        My = -self.T_B[0] * self.l / 2 + self.T_B[2] * self.l / 2
        Mz = (self.T_B[0] - self.T_B[1] + self.T_B[2] - self.T_B[3]) * self.c
        self.M_B = np.array([Mx, My, Mz])

        C_B_I = dir_cosine(self.q)  # inertial to body
        C_I_B = np.transpose(C_B_I)

        dr_I = self.v_I
        dv_I = 1 / self.m * np.dot(C_I_B, self.thrust_B) + self.g_I
        dq = 1 / 2 * np.dot(omega(self.w_B), self.q)
        dw = np.dot(np.transpose(self.J_B), self.M_B - np.dot(np.dot(skew(self.w_B), self.J_B), self.w_B))

        self.X = np.hstack((self.r_I, self.v_I, self.q, self.w_B))
        self.U = self.T_B
        self.f = np.hstack((dr_I, dv_I, dq, dw))
        new_state = self.state + (self.f * self.dt)
        
        self.state = np.array(new_state, dtype=np.float32)

        # cost
        goal_r_I = np.array([0, 0, 0])
        self.cost_r_I = np.dot(self.r_I - goal_r_I, self.r_I - goal_r_I)     #TRY
        # goal velocity
        goal_v_I = np.array([0, 0, 0])
        self.cost_v_I = np.dot(self.v_I - goal_v_I, self.v_I - goal_v_I)
        # final attitude error
        goal_q = toQuaternion(0, [0, 0, 1])
        goal_R_B_I = dir_cosine(goal_q)
        R_B_I = dir_cosine(self.q)
        self.cost_q = np.trace(np.identity(3) - np.dot(np.transpose(goal_R_B_I), R_B_I))
        # auglar velocity cost
        goal_w_B = np.array([0, 0, 0])
        self.cost_w_B = np.dot(self.w_B - goal_w_B, self.w_B - goal_w_B)
        # the thrust cost
        self.cost_thrust = np.dot(self.T_B, self.T_B)

        # self.wr = 0.01
        # self.wv = 0.001
        # self.wq = 0.001
        # self.ww = 0.001
        self.path_cost = self.wr * self.cost_r_I + \
                          self.wv * self.cost_v_I + \
                          self.ww * self.cost_w_B + \
                          self.wq * self.cost_q + \
                        self.wthrust * self.cost_thrust
        reward = 1.5

        if self.cost_r_I <= 9 :         # 到达目的地
            reward = reward - 150

        elif  self.cost_r_I >= 350 :    # 距离偏离目的地
            reward = reward #+ 200
        
        if self.cost_r_I >= 350 :
            done = True

        elif self.cost_r_I <= 9:
             done = True

        else :
            done = False

        return self.state, -reward, done, {}

    
    def reset(self,*,seed: Optional[int] = None,return_info: bool = False,options: Optional[dict] = None):
        super().reset(seed=seed)
              
        if not return_info:
             r_state = np.random.randint(-10, 10, size=[3, ]) + np.random.random((3,))
             v_state = np.random.randint(-5, 5, size=[3, ]) + np.random.random((3,))
             q_state = np.random.randint(-5, 5, size=[4, ]) + np.random.random((4,))
             w_state = np.random.randint(-5, 5, size=[3, ]) + np.random.random((3,))
             self.state = np.hstack([r_state, v_state, q_state, w_state])
             return np.array(self.state, dtype=np.float32)
        else:
            self.state = np.array([-1.016053771972656250e+01, 9.500893592834472656e+00 ,-6.988329410552978516e+00 ,-7.709161639213562012e-01, -4.963303089141845703e+00 
            ,2.989432334899902344e+00 ,6.142377257347106934e-01 ,-1.587003827095031738e+00, -4.391743659973144531e+00 ,-1.713695526123046875e-01, -3.374008417129516602e+00 
            ,-4.510629177093505859e+00, 4.668846607208251953e+00])
            return       np.array([-1.016053771972656250e+01, 9.500893592834472656e+00 ,-6.988329410552978516e+00 ,-7.709161639213562012e-01, -4.963303089141845703e+00 
            ,2.989432334899902344e+00 ,6.142377257347106934e-01 ,-1.587003827095031738e+00, -4.391743659973144531e+00 ,-1.713695526123046875e-01, -3.374008417129516602e+00 
            ,-4.510629177093505859e+00, 4.668846607208251953e+00])
    
    def render(self, mode="human"):
        return self.r_I, self.state
 
    def seed(self):
        print("seed")
        pass

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    
def dir_cosine( q):
        C_B_I = np.array([
            [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
            [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
            [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)] ])
        # C_B_I = np.array([
        #     [1 - 2 * (q[2] ** 2 + q[3] ** 2),2 * (q[1] * q[2] - q[0] * q[3]),2 * (q[1] * q[3] + q[0] * q[2])],
        #     [2 * (q[1] * q[2] + q[0] * q[3]),1 - 2 * (q[1] ** 2 + q[3] ** 2),2 * (q[2] * q[3] - q[0] * q[1])],
        #     [2 * (q[1] * q[3] - q[0] * q[2]),2 * (q[2] * q[3] + q[0] * q[1]),1 - 2 * (q[1] ** 2 + q[2] ** 2)]])
        return C_B_I


def omega( w):
        omeg = np.array([[0, -w[0], -w[1], -w[2]],
                         [w[0], 0, w[2], -w[1]],
                         [w[1], -w[2], 0, w[0]],
                         [w[2], w[1], -w[0], 0]])
        return omeg


def skew( v):
        v_cross = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])

        return v_cross


def toQuaternion(angle, dir):
        if type(dir) == list:
            dir = np.array(dir)
        dir = dir / (np.linalg.norm(dir)+0.00001)
        quat = np.zeros(4)
        quat[0] = math.cos(angle / 2)
        quat[1:] = math.sin(angle / 2) * dir
        return quat.tolist()



        
    

