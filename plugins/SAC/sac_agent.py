from math import gamma
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from plugins.SAC.buffer import ReplayBuffer
from plugins.SAC.actor_critic import Actor, CriticQ, CriticV
from torch.nn.utils.clip_grad import clip_grad_norm_


GAMMA = 0.995
TAU =5e-3
INITIAL_RANDOM_STEPS = 100
POLICY_UPDATE_FREQUENCE = 1

BUFFER_SIZE = 1000000
BATCH_SIZE = 256

LR_A = 3e-4
LR_Q = 3e-4

N_NEURONS = 256


class SAC:
    def __init__(self, action_dim, state_dim, model_path,test=False, alpha = LR_A, beta = LR_Q, gamma = GAMMA, tau = TAU,
                n_neurons = N_NEURONS, buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE):                
        self.statedim = state_dim
        self.actiondim = action_dim

        self.model_path = model_path

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.n_neurons = n_neurons
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.qf1_lossarr = np.array([])
        self.qf2_lossarr = np.array([])

        self.memory = ReplayBuffer(self.statedim,self.actiondim, self.buffer_size, self.batch_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.target_alpha = -np.prod((self.actiondim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = Actor(self.statedim, self.actiondim, test, hidden_dim1=self.n_neurons,hidden_dim2=self.n_neurons).to(self.device)

        self.vf = CriticV(self.statedim, hidden_dim1=self.n_neurons,hidden_dim2=self.n_neurons).to(self.device)
        self.vf_target = CriticV(self.statedim, hidden_dim1=self.n_neurons,hidden_dim2=self.n_neurons).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        self.qf1 = CriticQ(self.statedim + self.actiondim, hidden_dim1=self.n_neurons,hidden_dim2=self.n_neurons).to(self.device)
        self.qf2 = CriticQ(self.statedim + self.actiondim, hidden_dim1=self.n_neurons,hidden_dim2=self.n_neurons).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.beta)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=self.beta)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=self.beta)

        self.transition = [[]]

        self.total_step = 0

        self.is_test = False
        
        if self.device.type == 'cpu':
            print('DEVICE USED', self.device.type)
        else:
            print('DEVICE USED', torch.cuda.device(torch.cuda.current_device()), torch.cuda.get_device_name(0))

    def step(self, state):      
        selected_action = []
        action = self.actor(torch.FloatTensor(state).to(self.device))[0].detach().cpu().numpy()
        selected_action.append(action)
        selected_action = np.array(selected_action)
        selected_action = np.clip(selected_action, -1, 1)

        self.total_step += 1
        return selected_action.tolist()
    
    def store_transition(self,state, action, reward, new_state, done):       
        if not self.is_test:           
            self.transition = [state, action, reward, new_state, done]
            self.memory.store(*self.transition)
            
    def train(self):
        if (len(self.memory) >  BATCH_SIZE and self.total_step > INITIAL_RANDOM_STEPS):
            self.update_model()

    def update_model(self):
        device = self.device

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, self.actiondim)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        new_action, log_prob = self.actor(state)

        alpha_loss = ( -self.log_alpha.exp() * (log_prob + self.target_alpha).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        mask = 1 - done
        q1_pred = self.qf1(state, action)
        q2_pred = self.qf2(state, action)
        vf_target = self.vf_target(next_state)
        q_target = reward + self.gamma * vf_target * mask
        qf1_loss = F.mse_loss(q_target.detach(), q1_pred)
        qf2_loss = F.mse_loss(q_target.detach(), q2_pred)

        if len(self.qf1_lossarr) < 30000:
            self.qf1_lossarr = np.append(self.qf1_lossarr,qf1_loss.detach().cpu().numpy())
            self.qf2_lossarr = np.append(self.qf2_lossarr,qf2_loss.detach().cpu().numpy())

        v_pred = self.vf(state)
        q_pred = torch.min(
            self.qf1(state, new_action), self.qf2(state, new_action)
        )
        v_target = q_pred - alpha * log_prob
        v_loss = F.mse_loss(v_pred, v_target.detach())

        if self.total_step % POLICY_UPDATE_FREQUENCE== 0:
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)
        
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        qf_loss = qf1_loss + qf2_loss

        self.vf_optimizer.zero_grad()
        v_loss.backward()
        self.vf_optimizer.step()

        return actor_loss.data, qf_loss.data, v_loss.data, alpha_loss.data
    
    def save_models(self):
        torch.save(self.actor.state_dict(), self.model_path+"/model/actor.pt")
        torch.save(self.qf1.state_dict(),  self.model_path+"/model/qf1.pt")
        torch.save(self.qf2.state_dict(), self.model_path+"/model/qf2.pt")
        torch.save(self.vf.state_dict(),  self.model_path+"/model/vf.pt")       

    def load_models(self):
        # The models were trained on a CUDA device
        # If you are running on a CPU-only machine, use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
        self.actor.load_state_dict(torch.load( self.model_path+"/model/actor.pt", map_location=torch.device('cpu')))
        self.qf1.load_state_dict(torch.load( self.model_path+"/model/qf1.pt", map_location=torch.device('cpu')))
        self.qf2.load_state_dict(torch.load( self.model_path+"/model/qf2.pt", map_location=torch.device('cpu')))
        self.vf.load_state_dict(torch.load( self.model_path+"/model/vf.pt", map_location=torch.device('cpu')))
    
    def _target_soft_update(self):
        for t_param, l_param in zip(
            self.vf_target.parameters(), self.vf.parameters()
        ):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)

    def normalizeState(self, s_t):
        return s_t