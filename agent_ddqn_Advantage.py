#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import namedtuple, deque
import os
import sys

import wandb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from agent import Agent
from dqn_model import DQN

import matplotlib.pyplot as plt
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

BufferData = namedtuple('BufferData',('state', 'action', 'next_state', 'reward','done'))

class RainbowDQn(nn.Module):
    def __init__(self,in_dims,n_actions):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dims[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU())
        

        a = self.conv(Variable(torch.zeros(in_dims))).view(1, -1).size(1)

        self.fc1 = nn.Linear(a, 512)
        #self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batchsize = 32 # args.batchsize or 128
        self.gamma = 0.99
        self.epsilon_max = 1.0
        self.epsilon_min = 0.02
        buffersize = 10000
        learning_rate = 1.5e-4
        self.tau = 0.5
        self.net_update_count = 250
        self.step_count = 0
        self.loop_count = 0
        self.n_episodes = 400000 # args.n_episodes or 1000
        self.eps_decay_rate = self.n_episodes/50
        self.env = env
        self.n_actions = env.get_action_space().n
        self.policy_net = RainbowDQn((4,84,84),self.n_actions).to(self.device)
        self.target_net = RainbowDQn((4,84,84),self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.update_count = 0
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque([],maxlen=buffersize)

        wandb.init(project="Project3-Hope", entity="loser-rl")
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.step_count = 0
        self.loop_count = 0
        self.update_count = 0
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        sel = random.random()
        exp_fac = math.exp(-1*self.step_count/self.eps_decay_rate)
        eps = self.epsilon_min + (self.epsilon_max - self.epsilon_min)*exp_fac
        if sel > eps or test or self.step_count%500 == 0:
            policy_out = self.policy_net(observation)
            action_idx = torch.argmax(policy_out,dim=1)[0]
            action = action_idx.detach().item()
        else:
            return random.randrange(self.n_actions)
        ###########################
        return action
    
    def push(self,*args):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.memory.append(BufferData(*args))
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        samples = []
        for i in range(self.batchsize):
            idx = random.randrange(len(self.memory))
            samples.append(self.memory[idx])
            if idx<0.2*(len(self.memory)):
                del self.memory[idx]
        ###########################
        return samples
        
    def optimize_step(self):
        if(len(self.memory) < 10*self.batchsize): # or self.loop_count<2 :
            return
        self.loop_count = 0
        sample = self.replay_buffer()
        batch = BufferData(*zip(*sample))

        next_states = torch.cat(batch.next_state)
        states = torch.cat(batch.state)
        rewards = torch.cat(batch.reward)
        actions = torch.cat(batch.action)
        done = torch.cat(batch.done)
        states.to(self.device)
        next_states.to(self.device)

       # state_action_values = torch.zeros(self.batchsize, device=self.device)
       # for i in range(self.batchsize):
        #    state_action_values[i] = self.policy_net(torch.from_numpy(states[i])).gather(1,torch.from_numpy(actions[i]))

        #next_state_values = torch.zeros(self.batchsize, device=self.device)
        #with torch.no_grad():
        #    next_state_values[non_final_mask] = self.target_net(torch.from_numpy(next_states[non_final_mask])).max(1)[0]
        #expected_state_action_values = (next_state_values * self.gamma) + rewards
        intm_val = self.policy_net(states)
        #print(intm_val.shape)
        state_action_values = intm_val[torch.arange(intm_val.size(0)),actions]#torch.zeros(self.batchsize, device=self.device)
        #for i in range(self.batchsize):
            #state_action_values[i] = intm_val[i][actions[i].item()].item()
        #state_action_values = Variable(state_action_values,requires_grad=True)
        #state_action_values = torch.reshape(state_action_values.unsqueeze(1),(1, self.batchsize))[0]
        
        next_state_values = torch.zeros(self.batchsize, device=self.device)
        # with torch.no_grad():
            # plcy_act = done*self.policy_net(next_states).max(1)[1]
            # trgt_out = self.target_net(next_states)
        next_state_values = done*self.target_net(next_states).max(1)[0].detach()
            # for i in range(self.batchsize):
               # next_state_values[i] = done[i]*trgt_out[i].max(1)[0]
                # next_state_values[i] = done[i]*trgt_out[i][plcy_act[i].item()].item()
        
        expected_state_action_values = (next_state_values * self.gamma) + rewards
        #expected_state_action_values = Variable(expected_state_action_values,requires_grad=True)
        expected_state_action_values = torch.reshape(expected_state_action_values.unsqueeze(1),(1, self.batchsize))[0]
        # expected_state_action_values = Variable(expected_state_action_values,requires_grad=True)
        #print(state_action_values.shape)
        #print(expected_state_action_values.shape)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        #self.update_count += 1
        #if self.update_count%self.net_update_count==0:
            # target_net_state_dict = self.target_net.state_dict()
            # policy_net_state_dict = self.policy_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            # self.target_net.load_state_dict(target_net_state_dict)
            # self.target_net.load_state_dict(self.policy_net.state_dict())
            #self.update_count=1



    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        episode_time = []
        episode_reward = []
        ep_r = 0
        ep_t = 0
        for ne in range(self.n_episodes):
            self.step_count+=1
            start_state = self.env.reset()
            #print('New Episode ',ne)
            state = torch.from_numpy(np.transpose(np.asarray(start_state,dtype=np.float32)/255,(2,0,1))).unsqueeze(0)
            state = state.to(self.device)
            done = False
            queue_size = 5
            queue = deque([],maxlen=queue_size)
            tdn_reward = 0
            while not done:
                a = self.make_action(state,False)
                next_state, r, done, _, _ = self.env.step(a)
                ep_r += r
                tdn_reward += r*(self.gamma**len(queue))
                a_tensor = torch.tensor([a], dtype=torch.int64, device=self.device)
                # queue.append((state,a_tensor,r))
                if done:
                    n_state = state
                else:
                    n_state = torch.from_numpy(np.transpose(np.asarray(next_state,dtype=np.float32)/255,(2,0,1))).unsqueeze(0)
                n_state = n_state.to(self.device)
                done_tensor = torch.tensor([1-done], dtype=torch.int64,device=self.device)
                r_tensor = torch.tensor([r], dtype=torch.float32, device=self.device)
                self.push(state,a_tensor,n_state,r_tensor,done_tensor)
                # if len(queue)==queue_size:
                #     state_tensor,a_old_tensor,r_old = queue.popleft()
                #     r_tensor = torch.tensor([tdn_reward], dtype=torch.float32, device=self.device)
                #     self.push(state_tensor,a_old_tensor,n_state,r_tensor,done_tensor)
                #     tdn_reward = (tdn_reward - r_old)/self.gamma
                state = n_state
                self.optimize_step()
                self.loop_count+=1
                ep_t += 1
            # while len(queue)>0:
            #     state_tensor,a_old_tensor,r_old = queue.popleft()
            #     tdn_reward -= r_old
            #     tdn_reward /= self.gamma
            #     r_tensor = torch.tensor([tdn_reward], dtype=torch.float32, device=self.device)
            #     self.push(state_tensor,a_old_tensor,n_state,r_tensor,done_tensor)
            if ne%100==0:
                #print(ep_r/20)
                episode_reward.append(ep_r/100)
                exp_fac = math.exp(-1*self.step_count/self.eps_decay_rate)
                epsilon_val = self.epsilon_min + (self.epsilon_max - self.epsilon_min)*exp_fac
                wandb.log({"episode":ne,"epsilon":epsilon_val,"episode_reward": ep_r/100})
                ep_r = 0
            if ne%1000==0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if ne%5000==0:
                model_save_path = "models/DDQN_av_%d.model" %(ne)
                try:
                    torch.save(self.target_net.state_dict(),model_save_path)
                except:
                    print("some issue with saving")
            if ne%10==0:
                episode_time.append(ep_t/10)
                wandb.log({"episode":ne,"episode_time": ep_t/10})
                ep_t = 0
            
        
        plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
        plt.xlabel('Episodes')
        plt.ylabel('Time')
        plt.plot(episode_time)
        plt.savefig("time.png")
        plt.close()
        plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
        plt.xlabel('Episodes*100')
        plt.ylabel('Reward')
        plt.plot(episode_reward)
        plt.savefig("reward.png")
        plt.close()
        ###########################
