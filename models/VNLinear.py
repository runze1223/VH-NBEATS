import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual

        self.activations = {'relu': nn.ReLU(),
                    'softplus': nn.Softplus(),
                    'tanh': nn.Tanh(),
                    'selu': nn.SELU(),
                    'lrelu': nn.LeakyReLU(),
                    'prelu': nn.PReLU(),
                    'sigmoid': nn.Sigmoid()}

        if self.individual:
            self.mu = nn.ModuleList()
            for i in range(self.channels):
                self.mu.append(nn.Linear(self.seq_len,self.seq_len))
        else:
            self.mu=nn.Linear(self.seq_len,self.seq_len)
            

        if self.individual:
            self.var = nn.ModuleList()
            for i in range(self.channels):
                self.var.append(nn.Linear(self.seq_len,self.seq_len))
        else:
            hidden_layers = []
            hidden_layers.append(nn.Linear(self.seq_len, self.seq_len))
            hidden_layers.append(self.activations['selu'])
            hidden_layers.append(nn.InstanceNorm1d(num_features=self.seq_len))
            layers = hidden_layers
            self.var= nn.Sequential(*layers)
            
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:

            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        if self.individual:
            self.back = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.seq_len))
        else:
            self.back = nn.Linear(self.seq_len, self.seq_len)

    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def vae_loss(self,mu, logvar):
        KL_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return KL_loss

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last

        if self.individual:
            mu = torch.zeros([x.size(0),self.seq_len,x.size(2)],dtype=x.dtype).to(x.device)
            log_var = torch.zeros([x.size(0),self.seq_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                mu[:,:,i] = self.mu(x[:,:,i])
                log_var[:,:,i] = self.var(x[:,:,i])
            back = self.sample(mu, log_var)
            kl_divergence=self.vae_loss(mu,log_var)
            
        else:
            mu = self.mu(x.permute(0,2,1)).permute(0,2,1)
            log_var = self.var(x.permute(0,2,1)).permute(0,2,1)
            back = self.sample(mu, log_var)
            kl_divergence=self.vae_loss(mu,log_var)

            reproduce=self.back(back.permute(0,2,1)).permute(0,2,1)

        backward= reproduce+seq_last

        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](back[:,:,i])
        else:
            output = self.Linear(back.permute(0,2,1)).permute(0,2,1)
        output = output + seq_last

        return output,backward,kl_divergence # [Batch, Output length, Channel]



