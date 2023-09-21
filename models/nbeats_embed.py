# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:20:30 2022

@author: dell
"""
import math
import numpy as np
import torch 
import torch.nn as nn
from typing import Tuple
from layers.tcn import TemporalConvNet

import matplotlib.pyplot as plt

from layers.RevIN import RevIN

import copy

import numpy as np 

class BinaryConcrete(nn.Module):
    def __init__(self, temp,batch_size,ndim):
        super(BinaryConcrete, self).__init__()
        self.gumbel = torch.distributions.Gumbel(
            torch.zeros([batch_size,ndim,1]), torch.ones([batch_size,ndim,1]))
        self.temp = temp
        self.sigmoid = nn.Sigmoid()

    def forward(self,alpha):
        noise = torch.rand_like(alpha).cuda()
        noise=torch.log(noise)-torch.log(1-noise)
        ouput=self.sigmoid((alpha + noise) / self.temp)
        # ouput=self.sigmoid((alpha + self.gumbel.sample().cuda()) / self.temp)
        return ouput
    

class Concrete(nn.Module):
    def __init__(self, temp,batch_size,ndim):
        super(Concrete, self).__init__()
        self.gumbel = torch.distributions.Gumbel(
            torch.zeros([batch_size,ndim,2]), torch.ones([batch_size,ndim,2]))  
        self.temp = temp
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,alpha):
        return self.softmax((alpha + self.gumbel.sample().cuda()) / self.temp)
    

class Inference(nn.Module):
    def __init__(self, input=336, hidden=168, output=2,dim=7,individual=False):
        super(Inference, self).__init__()
        self.dim=dim
        self.individual=individual
        if individual:
            self.fc1 = nn.ModuleList()
            self.fc2 = nn.ModuleList()
            self.act_fn = nn.Tanh()
            for i in range(self.dim):
                self.fc1.append( nn.Linear(input, hidden))
                self.fc2.append(nn.Linear(hidden, output))

        else:
            self.fc1 = nn.Linear(input, hidden)
            self.fc2 = nn.Linear(hidden, output)
            self.sigmoid=nn.Sigmoid()
            self.act_fn = nn.Tanh()


    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.dim):
                z = self.fc1[i](x[:,i,:])          # z: [bs x d_model * patch_num]
                z = self.act_fn(z)
                z = self.fc2[i](z)                    # z: [bs x target_window]
                x_out.append(z)
            h = torch.stack(x_out, dim=1) 
        else:
            h = self.fc1(x)
            h = self.act_fn(h)
            h = self.fc2(h)
            h=self.sigmoid(h)
            h_new=torch.log(h+1e-08*torch.ones_like(h)/(torch.ones_like(h)-h+1e-08*torch.ones_like(h)))
            # h_new=torch.log(h/(torch.ones_like(h)-h+1e-08*torch.ones_like(h)))

        return h_new,h

############Year_BASIS########################
class Year_basis(nn.Module):
    def __init__(self, backcast_size, forecast_size,dim,batch_size,individual,beta):
        super(Year_basis,self).__init__()
        self.backcast_size=backcast_size
        self.forecast_size=forecast_size
        self.dim=dim
        self.temp=0.05
        self.inference = Inference(backcast_size, backcast_size//2, 1,dim,individual)
        self.batch_size=batch_size
        self.softmax = nn.Softmax(dim=-1)
        self.beta=beta
        self.weight = nn.Parameter(torch.zeros(dim))

        self.binaryConcrete=BinaryConcrete(self.temp, self.batch_size,self.dim)

    def generate_the_basis_function(self,index,input_len,output_len,basis, repeating_period): 


            repeat_foward= output_len//repeating_period+1
            repeat_backward= input_len//repeating_period+1

            baisis1=basis[index+1:,:]
            baisis2=basis[:index+1,:]
            basis=torch.cat([baisis1,baisis2],axis=0)
            repeat_foward=torch.tile(basis, (repeat_foward,1))
            repeat_backward=torch.tile(basis, (repeat_backward,1))
            repeat_foward=repeat_foward[:output_len,:]
            repeat_backward=repeat_backward[-input_len:,:]
            # output_basis=np.concatenate([repeat_backward,repeat_foward],axis=0)
            output_basis_foward=repeat_foward.permute(1,0)
            output_basis_backward=repeat_backward.permute(1,0)
            mean=torch.mean(output_basis_backward,axis=1)
            mean=mean.unsqueeze(1)
            mean2=mean.repeat(1,self.backcast_size)
            mean1=mean.repeat(1,self.forecast_size)
            output_basis_foward=(output_basis_foward-mean1)
            output_basis_backward=(output_basis_backward-mean2)

            return output_basis_foward, output_basis_backward
    
    def encode(self, x):
        return self.inference(x)

    def sample(self, alpha, temp=None):

        if self.training:
            residual=self.binaryConcrete(alpha)
            return residual
        else:
            return(torch.sigmoid((alpha)/self.temp)> 0.5).float()#torch.distributions.OneHotCategorical(logits=alpha).sample()
       

    def vae_loss(self,alpha):
        ones=torch.ones_like(alpha).cuda()
        kl=alpha*torch.log((alpha+1e-5*ones)/0.5*ones)+(1-alpha)*torch.log((1-alpha+1e-5*ones)/0.5*ones).mean()
        kl=torch.mean(kl)
        return kl
    

    def forward(self, theta,x,y):  

        x=x[:,-1,:]
        batch_size,n_dim,_=theta.size()
        basis_function_foward=torch.zeros(batch_size,1,n_dim,self.forecast_size).cuda()
        basis_function_backward= torch.zeros(batch_size,1,n_dim,self.backcast_size).cuda()
        basis=torch.tensor(y[3]).cuda()
        a,b= basis.size()

        for j in range(batch_size):
            index=x[j,2]
            basis_function_foward[j,0,:,:],basis_function_backward[j,0,:,:]= self.generate_the_basis_function(index,self.backcast_size,self.forecast_size, basis,a)

        theta=theta-torch.mean(theta,axis=-1).unsqueeze(2)  
        ones=torch.ones_like(theta)
        input=ones-(theta-basis_function_backward.squeeze())**2/((theta)**2 +1e-06*ones)
        alpha = self.encode(input)[0]
        weight= self.weight.unsqueeze(0)
        weight=weight.unsqueeze(-1)
        weight=weight.repeat(self.batch_size,1,1)
        alpha=(1-self.beta)*alpha+self.beta*weight
        
        sample = self.sample(alpha, self.temp)
        kl_diveregence=self.vae_loss(self.encode(input)[1])
        backcast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_backward)
        forecast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_foward)
        return backcast, forecast, kl_diveregence


############WEEKDAY_BASIS########################

class Weekday_basis(nn.Module):
    def __init__(self, backcast_size, forecast_size,dim,batch_size,individual,beta):
        super(Weekday_basis,self).__init__()
        self.backcast_size=backcast_size
        self.forecast_size=forecast_size
        self.dim=dim
        self.temp=0.05
        self.inference = Inference(backcast_size, backcast_size//2, 1,dim,individual)
        self.batch_size=batch_size
        self.softmax = nn.Softmax(dim=-1)
        self.beta=beta
        self.weight = nn.Parameter(torch.zeros(dim))
        self.binaryConcrete=BinaryConcrete(self.temp, self.batch_size,self.dim)

    def generate_the_basis_function(self,index,input_len,output_len,basis, repeating_period): 
            repeat_foward= output_len//repeating_period+1
            repeat_backward= input_len//repeating_period+1
            baisis1=basis[index+1:,:]
            baisis2=basis[:index+1,:]
            basis=torch.cat([baisis1,baisis2],axis=0)
            repeat_foward=torch.tile(basis, (repeat_foward,1))
            repeat_backward=torch.tile(basis, (repeat_backward,1))
            repeat_foward=repeat_foward[:output_len,:]
            repeat_backward=repeat_backward[-input_len:,:]
            output_basis_foward=repeat_foward.permute(1,0)
            output_basis_backward=repeat_backward.permute(1,0)
            mean=torch.mean(output_basis_backward,axis=1)
            mean=mean.unsqueeze(1)
            mean2=mean.repeat(1,self.backcast_size)
            mean1=mean.repeat(1,self.forecast_size)
            output_basis_foward=(output_basis_foward-mean1)
            output_basis_backward=(output_basis_backward-mean2)

            return output_basis_foward, output_basis_backward
    
    def encode(self, x):
        return self.inference(x)

    def sample(self, alpha, temp=None):

        if self.training:
            residual=self.binaryConcrete(alpha)
            return residual
        else:
            return(torch.sigmoid((alpha)/self.temp)> 0.5).float()#torch.distributions.OneHotCategorical(logits=alpha).sample()
       

    def vae_loss(self,alpha):
        # alpha=torch.sigmoid(alpha/self.temp)
        ones=torch.ones_like(alpha).cuda()
        kl=alpha*torch.log((alpha+1e-5*ones)/0.5*ones)+(1-alpha)*torch.log((1-alpha+1e-5*ones)/0.5*ones).mean()
        kl=torch.mean(kl)
        return kl
    

    def forward(self, theta,x,y):  

        x=x[:,-1,:]
        batch_size,n_dim,_=theta.size()
        basis_function_foward=torch.zeros(batch_size,1,n_dim,self.forecast_size).cuda()
        basis_function_backward= torch.zeros(batch_size,1,n_dim,self.backcast_size).cuda()
        basis=torch.tensor(y[0]).cuda()
        a,b= basis.size()
        for j in range(batch_size):
            index=x[j,0]
            basis_function_foward[j,0,:,:],basis_function_backward[j,0,:,:]= self.generate_the_basis_function(index,self.backcast_size,self.forecast_size, basis,a)
        theta=theta-torch.mean(theta,axis=-1).unsqueeze(2)  
        ones=torch.ones_like(theta)
        input=ones-(theta-basis_function_backward.squeeze())**2/((theta)**2 +1e-06*ones)
        alpha = self.encode(input)[0]
        weight= self.weight.unsqueeze(0)
        weight=weight.unsqueeze(-1)
        weight=weight.repeat(self.batch_size,1,1)
        alpha=(1-self.beta)*alpha+self.beta*weight
        sample = self.sample(alpha, self.temp)
        kl_diveregence=self.vae_loss(self.encode(input)[1])
        backcast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_backward)
        forecast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_foward)
        return backcast, forecast, kl_diveregence

############WEEK_BASIS########################
class Week_basis(nn.Module):
    def __init__(self, backcast_size, forecast_size,dim,batch_size,individual,beta):
        super(Week_basis,self).__init__()
        self.backcast_size=backcast_size
        self.forecast_size=forecast_size
        self.dim=dim
        self.temp=0.05
        self.inference = Inference(backcast_size, backcast_size//2, 1,dim,individual)
        self.batch_size=batch_size
        self.softmax = nn.Softmax(dim=-1)
        self.beta=beta
        self.weight = nn.Parameter(torch.zeros(dim))
        self.binaryConcrete=BinaryConcrete(self.temp, self.batch_size,self.dim)

    def generate_the_basis_function(self,index,input_len,output_len,basis, repeating_period,beta): 
            repeat_foward= output_len//repeating_period+1
            repeat_backward= input_len//repeating_period+1
            baisis1=basis[index+1:,:]
            baisis2=basis[:index+1,:]
            basis=torch.cat([baisis1,baisis2],axis=0)
            repeat_foward=torch.tile(basis, (repeat_foward,1))
            repeat_backward=torch.tile(basis, (repeat_backward,1))
            repeat_foward=repeat_foward[:output_len,:]
            repeat_backward=repeat_backward[-input_len:,:]
            output_basis_foward=repeat_foward.permute(1,0)
            output_basis_backward=repeat_backward.permute(1,0)
            mean=torch.mean(output_basis_backward,axis=1)
            mean=mean.unsqueeze(1)
            mean2=mean.repeat(1,self.backcast_size)
            mean1=mean.repeat(1,self.forecast_size)    
            output_basis_foward=(output_basis_foward-mean1)
            output_basis_backward=(output_basis_backward-mean2)

            return output_basis_foward, output_basis_backward
    
    def encode(self, x):
        return self.inference(x)

    def sample(self, alpha, temp=None):

        if self.training:
            residual=self.binaryConcrete(alpha)
            return residual
        else:
            return(torch.sigmoid((alpha)/self.temp)> 0.5).float()#torch.distributions.OneHotCategorical(logits=alpha).sample()
       

    def vae_loss(self,alpha):
        # alpha=torch.sigmoid(alpha/self.temp)
        ones=torch.ones_like(alpha).cuda()
        kl=alpha*torch.log((alpha+1e-5*ones)/0.5*ones)+(1-alpha)*torch.log((1-alpha+1e-5*ones)/0.5*ones).mean()
        kl=torch.mean(kl)
        return kl
    

    def forward(self, theta,x,y):  

        x=x[:,-1,:]
        batch_size,n_dim,_=theta.size()
        basis_function_foward=torch.zeros(batch_size,1,n_dim,self.forecast_size).cuda()
        basis_function_backward= torch.zeros(batch_size,1,n_dim,self.backcast_size).cuda()

        basis=torch.tensor(y[1]).cuda()
        a,b= basis.size()

        for j in range(batch_size):
            index=x[j,0]
            basis_function_foward[j,0,:,:],basis_function_backward[j,0,:,:]= self.generate_the_basis_function(index,self.backcast_size,self.forecast_size, basis,a)

        theta=theta-torch.mean(theta,axis=-1).unsqueeze(2)      
        ones=torch.ones_like(theta)
        input=ones-(theta-basis_function_backward.squeeze())**2/((theta)**2 +1e-06*ones)
        alpha = self.encode(input)[0]
        weight= self.weight.unsqueeze(0)
        weight=weight.unsqueeze(-1)
        weight=weight.repeat(self.batch_size,1,1)
        alpha=(1-self.beta)*alpha+self.beta*weight
        sample = self.sample(alpha, self.temp)
        kl_diveregence=self.vae_loss(self.encode(input)[1])
        backcast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_backward)
        forecast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_foward)
        return backcast, forecast, kl_diveregence

############Day_BASIS########################
class Day_basis(nn.Module):
    def __init__(self, backcast_size, forecast_size,dim,batch_size,individual,beta):
        super(Day_basis,self).__init__()
        self.backcast_size=backcast_size
        self.forecast_size=forecast_size
        self.dim=dim
        self.temp=0.1
        self.inference = Inference(backcast_size, backcast_size//2, 1,dim,individual)
        self.batch_size=batch_size
        self.softmax = nn.Softmax(dim=-1)

        self.weight = nn.Parameter(torch.zeros(dim))
        self.binaryConcrete=BinaryConcrete(self.temp, self.batch_size,self.dim)
        self.beta=beta

    def generate_the_basis_function(self,index,input_len,output_len,basis, repeating_period): 
            repeat_foward= output_len//repeating_period+1
            repeat_backward= input_len//repeating_period+1

            baisis1=basis[index+1:,:]
            baisis2=basis[:index+1,:]
            basis=torch.cat([baisis1,baisis2],axis=0)

            repeat_foward=torch.tile(basis, (repeat_foward,1))
            repeat_backward=torch.tile(basis, (repeat_backward,1))

            repeat_foward=repeat_foward[:output_len,:]
            repeat_backward=repeat_backward[-input_len:,:]
            output_basis_foward=repeat_foward.permute(1,0)
            output_basis_backward=repeat_backward.permute(1,0)
            mean=torch.mean(output_basis_backward,axis=1)
            mean=mean.unsqueeze(1)
            mean2=mean.repeat(1,self.backcast_size)
            mean1=mean.repeat(1,self.forecast_size)

            output_basis_foward=(output_basis_foward-mean1)
            output_basis_backward=(output_basis_backward-mean2)

            return output_basis_foward, output_basis_backward
    
    def encode(self, x):
        return self.inference(x)

    def sample(self, alpha, temp=None):

        if self.training:
            residual=self.binaryConcrete(alpha)
            return residual
        else:
            return(torch.sigmoid((alpha)/self.temp)> 0.5).float()#torch.distributions.OneHotCategorical(logits=alpha).sample()


    def vae_loss(self,alpha):
        # alpha=torch.sigmoid(alpha/self.temp)
        ones=torch.ones_like(alpha).cuda()
        kl=alpha*torch.log((alpha+1e-5*ones)/0.5*ones)+(1-alpha)*torch.log((1-alpha+1e-5*ones)/0.5*ones).mean()
        kl=torch.mean(kl)

        return kl
    

    def forward(self, theta,x,y):  

        x=x[:,-1,:]
        batch_size,n_dim,_=theta.size()
        basis_function_foward=torch.zeros(batch_size,1,n_dim,self.forecast_size).cuda()
        basis_function_backward= torch.zeros(batch_size,1,n_dim,self.backcast_size).cuda()

        basis=torch.tensor(y[2]).cuda()
        a,b= basis.size()


        for j in range(batch_size):
            index=x[j,1]
            basis_function_foward[j,0,:,:],basis_function_backward[j,0,:,:]= self.generate_the_basis_function(index,self.backcast_size,self.forecast_size, basis,a)
        theta=theta-torch.mean(theta,axis=-1).unsqueeze(2)      
        ones=torch.ones_like(theta)
        input=ones-(theta-basis_function_backward.squeeze())**2/((theta)**2 +1e-06*ones)
        alpha = self.encode(input)[0]
        weight= self.weight.unsqueeze(0)
        weight=weight.unsqueeze(-1)
        weight=weight.repeat(self.batch_size,1,1)
        alpha=(1-self.beta)*alpha+self.beta*weight
        sample = self.sample(alpha, self.temp)
        kl_diveregence=self.vae_loss(self.encode(input)[1])

        backcast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_backward)
        forecast = torch.einsum('bkp,bpkt->bkt', sample, basis_function_foward)

        return backcast, forecast, kl_diveregence


class TrendBasis(nn.Module):
    def __init__(self, degree_of_polynomial, backcast_size, forecast_size,variation):
        super(TrendBasis,self).__init__()
        polynomial_size = degree_of_polynomial + 1
        self.backcast_size=backcast_size
        self.forecast_size=forecast_size           
        total_grid=torch.tensor(np.concatenate([np.power(np.arange(forecast_size+backcast_size, dtype=np.float) / (forecast_size+backcast_size), i)[None, :]
                                    for i in range(polynomial_size)]), dtype=torch.float32)
        backcast_template=total_grid[:,:backcast_size]
        forecast_template=total_grid[:,backcast_size:]
        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)
        self.backcast_size=backcast_size
        self.forecast_size=forecast_size
    def forward(self, theta,x,y):  
        backcast = torch.einsum('bkp,pt->bkt', theta, self.backcast_basis) 
        forecast = torch.einsum('bkp,pt->bkt', theta, self.forecast_basis)
        return backcast, forecast

class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics, backcast_size, forecast_size,degree_of_polynomial,variation):
        super(SeasonalityBasis,self).__init__()
        self.backcast_size=backcast_size
        self.forecast_size=forecast_size
   
        frequency = np.append(np.zeros(1, dtype=np.float32),
                                        np.arange(harmonics, harmonics / 2 * (forecast_size+backcast_size),
                                                    dtype=np.float32)/ harmonics)[None, :]
        total_grid = -2 * np.pi * (
                np.arange(backcast_size+forecast_size, dtype=np.float32)[:, None] / (forecast_size+backcast_size)) * frequency

        total_grid2 = -2 * np.pi * (
                np.arange(backcast_size+forecast_size, dtype=np.float32)[:, None] / (forecast_size+backcast_size)) * frequency- -0.25 * np.pi* frequency

        self.backcast_size  =  backcast_size  

        backcast_grid=total_grid[:backcast_size,:]
        forecast_grid=total_grid[backcast_size:,:]
        backcast_grid2=total_grid2[:backcast_size,:]
        forecast_grid2=total_grid2[backcast_size:,:]
        
        backcast_cos_template = torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32)
        backcast_sin_template = torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32)
        backcast_cos_template2 = torch.tensor(np.transpose(np.cos(backcast_grid2)), dtype=torch.float32)
        backcast_sin_template2 = torch.tensor(np.transpose(np.sin(backcast_grid2)), dtype=torch.float32)

        backcast_template = torch.cat([backcast_cos_template, backcast_sin_template,backcast_cos_template2, backcast_sin_template2], dim=0)

        forecast_cos_template = torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32)
        forecast_sin_template = torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32)
        forecast_cos_template2 = torch.tensor(np.transpose(np.cos(forecast_grid2)), dtype=torch.float32)
        forecast_sin_template2 = torch.tensor(np.transpose(np.sin(forecast_grid2)), dtype=torch.float32)
        forecast_template = torch.cat([forecast_cos_template, forecast_sin_template,forecast_cos_template2, forecast_sin_template2], dim=0)

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)
    def forward(self, theta,x,y) :
        backcast = torch.einsum('bkp,pt->bkt', theta, self.backcast_basis)
        forecast = torch.einsum('bkp,pt->bkt', theta, self.forecast_basis)
        return backcast, forecast 

       
class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, x_t_n_inputs, x_s_n_inputs, x_s_n_hidden, theta_n_dim, basis: nn.Module,
                 n_layers, theta_n_hidden, include_var_dict, t_cols, batch_normalization, dropout_prob, activation,variation,channel):
        """
        """
        super(NBeatsBlock,self).__init__()

        if x_s_n_inputs == 0:
            x_s_n_hidden = 0
        theta_n_hidden = [x_t_n_inputs + x_s_n_hidden] + theta_n_hidden
        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.variation=variation
        self.channel=channel
        self.activations = {'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}

        hidden_layers = []
        for i in range(n_layers):
            # Batch norm after activation
            hidden_layers.append(nn.Linear(in_features=theta_n_hidden[i], out_features=theta_n_hidden[i+1]))
            # hidden_layers.append(self.activations[activation])
            if i+1<n_layers:
                if self.batch_normalization:
                    hidden_layers.append(nn.InstanceNorm1d(num_features=theta_n_hidden[i+1]))
                if self.dropout_prob>0:
                    hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        layers = hidden_layers
        self.layers = nn.Sequential(*layers)

        hidden_layers = []
        for i in range(n_layers):
            # Batch norm after activation
            hidden_layers.append(nn.Linear(in_features=theta_n_hidden[i], out_features=theta_n_hidden[i+1]))     
            hidden_layers.append(self.activations[activation])
            if i<n_layers:
                if self.batch_normalization:
                    hidden_layers.append(nn.InstanceNorm1d(num_features=theta_n_hidden[i+1]))
                    # hidden_layers.append(nn.InstanceNorm1d(num_features=self.channel))
                # if self.dropout_prob>0:
                #     hidden_layers.append(nn.Dropout(p=self.dropout_prob))    
        layers = hidden_layers
        self.layers2 = nn.Sequential(*layers)
        self.output_layer=nn.Linear(in_features=theta_n_hidden[-1], out_features=theta_n_dim)
        self.basis = basis
        
    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def vae_loss(self,mu, logvar):
        KL_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return KL_loss
        

    def forward(self, insample_y,x,y) :

        # Compute local projection weights and 
        if self.variation==True:
            if self.training:
                mu = self.layers(insample_y)
                log_var = self.layers2(insample_y)
                ouput = self.sample(mu, log_var)
                kl_divergence=self.vae_loss(mu,log_var)
            else:
                mu = self.layers(insample_y)
                log_var = self.layers2(insample_y)
                std = torch.exp(0.5*log_var)
                ouput = mu
                log_var = None
                kl_divergence=torch.tensor([0]).float().cuda()
            theta=self.output_layer(ouput) 
        else:
            ouput=self.layers(insample_y)
            theta=self.output_layer(ouput)

        backcast, forecast = self.basis(theta,x,y)
        if  self.variation==True:
            return backcast, forecast,kl_divergence,theta
        else:
            return backcast, forecast
            
class NBeats(nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self,  x_t_n_inputs, x_s_n_inputs, x_s_n_hidden, n_polynomials,  n_harmonics,
                 n_layers, theta_n_hidden, include_var_dict, t_cols, batch_normalization, dropout_prob, output_size, activation,variation,channel,embedding,batch_size,embed,beta,duplicate,alpha_0,alpha_1):    
        super(NBeats,self).__init__()
        self.input_size=x_t_n_inputs
        self.n_hidden=theta_n_hidden
        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols
        self.batch_normalization = batch_normalization
        self.dropout_prob_theta = dropout_prob
        self.activation = activation
        self.n_layers=n_layers
        self.output_size=output_size
        self.n_polynomials=n_polynomials
        self.n_harmonics=n_harmonics
        self.variation=variation
        self.channel=channel
        self.embedding=embedding
        self.alpha_0=alpha_0
        self.alpha_1=alpha_1
        
        block_list=[]
        self.nbeats_block = NBeatsBlock(x_t_n_inputs = self.input_size,
                                    x_s_n_inputs =0,
                                    x_s_n_hidden= self.x_s_n_hidden,
                                    theta_n_dim=(self.n_polynomials + 1),
                                    basis=TrendBasis(degree_of_polynomial=self.n_polynomials,
                                                            backcast_size=self.input_size,
                                                            forecast_size=self.output_size,
                                                            variation=self.variation).cuda(),
                                    n_layers=self.n_layers,
                                    theta_n_hidden=[ 2*(self.n_polynomials + 1)],
                                    include_var_dict=self.include_var_dict,
                                    t_cols=self.t_cols,
                                    batch_normalization=self.batch_normalization,
                                    dropout_prob=self.dropout_prob_theta,
                                    activation=self.activation,
                                    variation=self.variation,
                                    channel=self.channel)
        block_list.append(self.nbeats_block)

        self.nbeats_block2 = NBeatsBlock(x_t_n_inputs = self.input_size,
                x_s_n_inputs = 0,
                x_s_n_hidden= self.x_s_n_hidden,
                theta_n_dim= 4 * int(
                        np.ceil(self.n_harmonics / 2 * (self.output_size+self.input_size)) - (self.n_harmonics - 1)),
                basis=SeasonalityBasis(harmonics=self.n_harmonics,
                                        backcast_size=self.input_size,
                                        forecast_size=self.output_size,
                                        degree_of_polynomial=self.n_polynomials,
                                        variation=self.variation).cuda(),
                n_layers=self.n_layers,
                theta_n_hidden= [4 * int(
                        np.ceil(self.n_harmonics / 2 * (self.output_size+self.input_size)) - (self.n_harmonics - 1))],
                include_var_dict=self.include_var_dict,
                t_cols=self.t_cols,
                batch_normalization=self.batch_normalization,
                dropout_prob=self.dropout_prob_theta,
                activation=self.activation,
                variation=self.variation,
                channel=self.channel)
        block_list.append(self.nbeats_block2)


        if duplicate==True:
            self.nbeats_block3 = NBeatsBlock(x_t_n_inputs = self.input_size,
                x_s_n_inputs = 0,
                x_s_n_hidden= self.x_s_n_hidden,
                theta_n_dim= 4 * int(
                        np.ceil(self.n_harmonics / 2 * (self.output_size+self.input_size)) - (self.n_harmonics - 1)),
                basis=SeasonalityBasis(harmonics=self.n_harmonics,
                                        backcast_size=self.input_size,
                                        forecast_size=self.output_size,
                                        degree_of_polynomial=self.n_polynomials,
                                        variation=self.variation).cuda(),
                n_layers=self.n_layers,
                theta_n_hidden= [4 * int(
                        np.ceil(self.n_harmonics / 2 * (self.output_size+self.input_size)) - (self.n_harmonics - 1))],
                include_var_dict=self.include_var_dict,
                t_cols=self.t_cols,
                batch_normalization=self.batch_normalization,
                dropout_prob=self.dropout_prob_theta,
                activation=self.activation,
                variation=self.variation,
                channel=self.channel)
        
            block_list.append(self.nbeats_block3)
            
        self.blocks= block_list
        self.basis_list=[]
        if 0 in self.embedding:
            self.vae1=Year_basis(self.input_size,self.output_size,self.channel,batch_size,embed,beta)
            self.basis_list.append(self.vae1)
        if 1 in self.embedding:
            self.vae2=Weekday_basis(self.input_size,self.output_size,self.channel,batch_size,embed,beta)
            self.basis_list.append(self.vae2)
        if 2 in self.embedding:
            self.vae3=Week_basis(self.input_size,self.output_size,self.channel,batch_size,embed,beta)
            self.basis_list.append(self.vae3)
        if 3 in self.embedding:
            self.vae4=Day_basis(self.input_size,self.output_size,self.channel,batch_size,embed,beta)
            self.basis_list.append(self.vae4)
    def forward(self, insample_y,x,y):
        insample_y= insample_y.permute(0,2,1)
        mean_i=torch.mean(insample_y,axis=2)
        insample_y=insample_y-mean_i.unsqueeze(2)
        kl_divergence_total=[]
        kl_divergence_total2=[]

        forecast_back=mean_i.unsqueeze(2)
        forecast=mean_i.unsqueeze(2)
        for i in range(len(self.basis_list)):
            basis_back,basis_foward,kl_divergence=self.basis_list[i](insample_y,x,y)
            insample_y = insample_y-basis_back
            forecast_back=forecast_back+basis_back
            forecast=forecast+basis_foward
            kl_divergence_total2.append(kl_divergence)

        new_back=torch.zeros_like(forecast_back)
        new_fore=torch.zeros_like(forecast)
        residuals = insample_y
        block_forecasts = []
        if self.variation==True:
            for i, block in enumerate(self.blocks):
                backcast, block_forecast,kl_divergence,theta = block(residuals,x,y)
                residuals = (residuals - backcast) 
                forecast = forecast + block_forecast#*forecast_var2
                new_back=new_back+backcast
                new_fore=new_fore+block_forecast
                kl_divergence_total.append(kl_divergence)

            kl_divergence_total=torch.stack(kl_divergence_total)
            kl_divergence_total2=torch.stack(kl_divergence_total2)
            kl_divergence_total=self.alpha_0*torch.mean(kl_divergence_total,axis=0)+self.alpha_1*torch.mean(kl_divergence_total2,axis=0)
            return forecast, forecast_back, kl_divergence_total            
        else:
            for i, block in enumerate(self.blocks):

                backcast, block_forecast = block(residuals,x,y)            
                forecast_back = forecast_back + backcast

                residuals = (residuals - backcast) 
                forecast = forecast + block_forecast
                block_forecasts.append(block_forecast)

            return forecast, forecast_back

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        
        self.input_size=configs.seq_len
        self.n_hidden=configs.theta_n_hidden
        self.x_s_n_inputs = 0
        self.x_s_n_hidden = 0
        self.include_var_dict = None
        self.t_cols = None
        self.batch_normalization = configs.batch_normalization
        self.dropout_prob_theta = configs.dropout_prob
        self.activation = configs.activation_nbeats
        self.n_layers=configs.n_layers
        self.output_size=configs.pred_len
        self.n_polynomials=configs.n_polynomials
        self.n_harmonics=configs.n_harmonics
        self.channel=configs.enc_in
        self.individual=configs.individual
        self.variation=configs.variation
        self.batch_size=configs.batch_size
        individual_embed=configs.individual_embed
        self.beta=configs.beta

        duplicate=configs.duplicate
        alpha_0=configs.alpha_0
        alpha_1=configs.alpha_1

        self.activations = {'relu': nn.ReLU(),
                    'softplus': nn.Softplus(),
                    'tanh': nn.Tanh(),
                    'selu': nn.SELU(),
                    'lrelu': nn.LeakyReLU(),
                    'prelu': nn.PReLU(),
                    'sigmoid': nn.Sigmoid()}

        self.embedding=configs.embedding

        if self.individual==True:
            self.nbeats_together = nn.ModuleList()
            for i in range(self.channel):
                self.nbeats_together.append(NBeats(self.input_size, self.x_s_n_inputs,self.x_s_n_hidden,self.n_polynomials, self.n_harmonics,
                              self.n_layers, self.n_hidden, self.include_var_dict, self.t_cols, self.batch_normalization, self.dropout_prob_theta, self.output_size, self.activation,self.variation,self.channel,self.embedding,self.batch_size,individual_embed,self.beta,duplicate,alpha_0,alpha_1))
        else:
            
            self.nbeats=NBeats(self.input_size, self.x_s_n_inputs,self.x_s_n_hidden,self.n_polynomials, self.n_harmonics,
                          self.n_layers, self.n_hidden, self.include_var_dict, self.t_cols, self.batch_normalization, self.dropout_prob_theta, self.output_size, self.activation,self.variation,self.channel,self.embedding,self.batch_size,individual_embed,self.beta,duplicate,alpha_0,alpha_1)   

        
    def forward(self, x_enc,x,y):

        mean_i=torch.mean(x_enc,axis=1)

        if self.individual==True:
            if self.variation==True:
                forecast_total=[]
                forecastback_total=[]
                kl_divergence_total=[]
                for i in range(self.channel):  
                    x_input=x_enc[:,:,i]
                    forecast,forecastback, kl_divergence=self.nbeats_together[i](x_input,x,y)
                    forecast_total.append(forecast)
                    forecastback_total.append(forecastback)
                    kl_divergence_total.append(kl_divergence)

                forecast_total=torch.stack(forecast_total,axis=2)
                forecastback_total=torch.stack(forecastback_total,axis=2)
                kl_divergence_total=torch.mean(t.stack(kl_divergence_total,axis=0))
            else:
                forecast_total=[]
                forecastback_total=[]
                for i in range(self.channel):            
                    x_input=x_enc[:,:,i]
                    forecast,forecastback=self.nbeats_together[i](x_input,x,y)
                    forecast_total.append(forecast)
                    forecastback_total.append(forecastback)
                forecast_total=torch.stack(forecast_total,axis=2)
                forecastback_total=torch.stack(forecastback_total,axis=2)
            output=forecast_total

        else:
            if self.variation==True:
                x_input=x_enc
                forecast,forecastback, kl_divergence=self.nbeats(x_input,x,y)
            else:
                x_input=x_enc
                forecast,forecastback=self.nbeats(x_input,x,y)

            forecast=forecast.permute(0,2,1)
            forecastback=forecastback.permute(0,2,1)
        mean_a=torch.mean(forecastback,axis=1)
        difference=mean_i-mean_a
        difference=difference.unsqueeze(1)
        if self.variation==True:
            return forecast,forecastback, kl_divergence# [B, L, D],
        else: 
            return forecast# [B, L, D],


