import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.nbeats import NBeats

class Nbeats_Multi(nn.Module):
    def __init__(self,x_t_n_inputs, x_s_n_inputs, x_s_n_hidden, n_polynomials,  n_harmonics,
                 n_layers, theta_n_hidden, include_var_dict, t_cols, batch_normalization, dropout_prob, output_size, activation,channel,individual):
        super(Nbeats_Multi, self).__init__()
        
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
        self.channel=channel
        self.individual=individual
        # self.linear_together = nn.ModuleList()
        if self.individual==True:
            self.nbeats_together = nn.ModuleList()
            for i in range(self.channel):
                self.nbeats_together.append(NBeats(x_t_n_inputs, x_s_n_inputs, x_s_n_hidden, n_polynomials,  n_harmonics,
                              n_layers, theta_n_hidden, include_var_dict, t_cols, batch_normalization, dropout_prob, output_size, activation))
        else:
            self.nbeats=NBeats(x_t_n_inputs, x_s_n_inputs, x_s_n_hidden, n_polynomials,  n_harmonics,
                          n_layers, theta_n_hidden, include_var_dict, t_cols, batch_normalization, dropout_prob, output_size, activation)
            
        # for i in range(self.channel):
        #     self.linear_together.append(nn.Linear(self.input_size, self.output_size))
            
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,variation=False,):
        
        
        if variation==True:
            forecast_total=[]
            forecastback_total=[]

            kl_divergence_total=[]
            for i in range(self.channel):            
                x_input=x_enc[:,:,i]
                if self.individual==True:
                    forecast,forecastback,kl_divergence=self.nbeats_together[i](x_input,variation)
                else:
                    forecast,forecastback,kl_divergence=self.nbeats(x_input,variation)

                forecast_total.append(forecast)
                forecastback_total.append(forecastback)
                kl_divergence_total.append(kl_divergence)
                
                # new=x_input-forecastback
                # new=self.linear_together[i](new)
                # forecast=forecast+new
                
            forecast_total=torch.stack(forecast_total,axis=2)
            forecastback_total=torch.stack(forecastback_total,axis=2)
        else:
            forecast_total=[]
            forecastback_total=[]

            for i in range(self.channel):            
                x_input=x_enc[:,:,i]
                forecast,forecastback=self.nbeats(x_input)
                forecast_total.append(forecast)
                forecastback_total.append(forecastback)
            forecast_total=torch.stack(forecast_total,axis=2)
            forecastback_total=torch.stack(forecastback_total,axis=2)
            
            
        output=forecast_total
            
        if variation==True:
            return output,forecastback_total,kl_divergence_total# [B, L, D],
        else:
            return output,forecastback_total# [B, L, D],


    def get_item(self, x_enc, x_mark_enc, x_dec, x_mark_dec, nbeats,step=2,variation=False,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        if variation==True:
            forecast_total=[]
            forecastback_total=[]

            kl_divergence_total=[]
            for i in range(self.channel):            
                x_input=x_enc[:,:,i]
                forecast,forecastback,kl_divergence,store1,store2=self.nbeats.get_item(x_input,variation)
                forecast_total.append(forecast)
                forecastback_total.append(forecastback)
                kl_divergence_total.append(kl_divergence)
     
            forecast_total=torch.stack(forecast_total,axis=2)
            forecastback_total=torch.stack(forecastback_total,axis=2)
        else:
            forecast_total=[]
            forecastback_total=[]

            for i in range(self.channel):            
                x_input=x_enc[:,:,i]
                forecast,forecastback=self.nbeats(x_input)
                forecast_total.append(forecast)
                forecastback_total.append(forecastback)
            forecast_total=torch.stack(forecast_total,axis=2)
            forecastback_total=torch.stack(forecastback_total,axis=2)
        output=forecast_total
        if variation==True:
            return output,forecastback_total,kl_divergence_total,store1,store2       # [B, L, D],
        else:
            return output,forecastback_total# [B, L, D],
        

# class Nbeats_Multi(nn.Module):
#     def __init__(self,x_t_n_inputs, x_s_n_inputs, x_s_n_hidden, n_polynomials,  n_harmonics,
#                  n_layers, theta_n_hidden, include_var_dict, t_cols, batch_normalization, dropout_prob, output_size, activation,channel):
#         super(Nbeats_Multi, self).__init__()
        
        
#         self.input_size=x_t_n_inputs
#         self.n_hidden=theta_n_hidden
#         self.x_s_n_inputs = x_s_n_inputs
#         self.x_s_n_hidden = x_s_n_hidden
#         self.include_var_dict = include_var_dict
#         self.t_cols = t_cols
#         self.batch_normalization = batch_normalization
#         self.dropout_prob_theta = dropout_prob
#         self.activation = activation
#         self.n_layers=n_layers
#         self.output_size=output_size
#         self.n_polynomials=n_polynomials
#         self.n_harmonics=n_harmonics
#         self.channel=channel
#         self.linear_together = nn.ModuleList()
        
#         self.nbeats=NBeats(x_t_n_inputs, x_s_n_inputs, x_s_n_hidden, n_polynomials,  n_harmonics,
#                      n_layers, theta_n_hidden, include_var_dict, t_cols, batch_normalization, dropout_prob, output_size, activation)
        
    
#         for i in range(self.channel):
#             self.linear_together.append(nn.Linear(self.input_size, self.output_size))
            
            
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,variation=False,):
        
        
#         if variation==True:
#             forecast_total=[]
#             forecastback_total=[]

#             kl_divergence_total=[]
#             for i in range(self.channel):            
#                 x_input=x_enc[:,:,i]
#                 forecast,forecastback,kl_divergence=self.nbeats(x_input,variation)
#                 forecast_total.append(forecast)
#                 forecastback_total.append(forecastback)
#                 kl_divergence_total.append(kl_divergence)
                
#                 new=x_input-forecastback
#                 new=self.linear_together[i](new)
#                 forecast=forecast+new
                
#             forecast_total=torch.stack(forecast_total,axis=2)
#             forecastback_total=torch.stack(forecastback_total,axis=2)
#         else:
#             forecast_total=[]
#             forecastback_total=[]
            
#             for i in range(self.channel):            
#                 x_input=x_enc[:,:,i]
#                 forecastbase = x_input[:, -1:]
#                 x_input=x_input-forecastbase
                
                
                
#                 forecast=self.linear_together[i](x_input)
#                 forecast=forecast+forecastbase   
#                 forecast_total.append(forecast)
             
                
#             forecast_total=torch.stack(forecast_total,axis=2)
            
#         output=forecast_total
            
#         if variation==True:
#             return output,forecastback_total,kl_divergence_total# [B, L, D],
#         else:
#             return output,output# [B, L, D],


#     def get_item(self, x_enc, x_mark_enc, x_dec, x_mark_dec, nbeats,step=2,variation=False,
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
#         if variation==True:
#             forecast_total=[]
#             forecastback_total=[]

#             kl_divergence_total=[]
#             for i in range(self.channel):            
#                 x_input=x_enc[:,:,i]
#                 forecast,forecastback,kl_divergence,store1,store2=self.nbeats_together[i].get_item(x_input,variation)
#                 forecast_total.append(forecast)
#                 forecastback_total.append(forecastback)
#                 kl_divergence_total.append(kl_divergence)
     
#             forecast_total=torch.stack(forecast_total,axis=2)
#             forecastback_total=torch.stack(forecastback_total,axis=2)
#         else:
#             forecast_total=[]
#             forecastback_total=[]

#             for i in range(self.channel):            
#                 x_input=x_enc[:,:,i]
#                 forecast,forecastback=self.nbeats_together[i](x_input)
#                 forecast_total.append(forecast)
#                 forecastback_total.append(forecastback)
#             forecast_total=torch.stack(forecast_total,axis=2)
#             forecastback_total=torch.stack(forecastback_total,axis=2)
#         output=forecast_total
#         if variation==True:
#             return output,forecastback_total,kl_divergence_total,store1,store2       # [B, L, D],
#         else:
#             return output,forecastback_total# [B, L, D],





              
            
# class Informer(nn.Module):
#     def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, seq_out, 
#                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
#                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
#                 output_attention = False, distil=True, mix=True,
#                 device=torch.device('cuda:0')):
#         super(Informer, self).__init__()

#         self.enc_in=enc_in
#         self.seq_len = seq_len
#         self.pred_len = seq_out
        
#         # Use this line if you want to visualize the weights
#         # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#         self.channels = enc_in
#         self.individual = True
#         if self.individual:
#             self.Linear = nn.ModuleList()
#             for i in range(self.channels):
#                 self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
#         else:
#             self.Linear = nn.Linear(self.seq_len, self.pred_len)
            
            
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, nbeats,step=2,variation=False,
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        
        
#         if variation==True:
#             forecast_total=[]
#             forecastback_total=[]
#             residual_total=[]
            
            
#             kl_divergence_total=[]

#             for i in range(self.enc_in):            
#                 x_input=x_enc[:,:,i]
#                 forecast,forecastback,kl_divergence=nbeats(x_input,variation)
#                 forecast_total.append(forecast)
#                 forecastback_total.append(forecastback)
#                 # residual_total.append(residual)
#                 kl_divergence_total.append(kl_divergence)
     

#             forecast_total=torch.stack(forecast_total,axis=2)
#             forecastback_total=torch.stack(forecastback_total,axis=2)
            
#         else:
#             forecast_total=[]
#             forecastback_total=[]
#             residual_total=[]
            

#             for i in range(self.enc_in):            
#                 x_input=x_enc[:,:,i]
#                 forecast,forecastback=nbeats(x_input)
#                 forecast_total.append(forecast)
#                 forecastback_total.append(forecastback)
#                 # residual_total.append(residual)

#             forecast_total=torch.stack(forecast_total,axis=2)
#             forecastback_total=torch.stack(forecastback_total,axis=2)
                        

#         # residual_total=torch.stack(residual_total,axis=2)
        
        
    
 
#         if step==2:
            
#             residual_total=x_enc-forecastback_total.detach()
#             if self.individual:
#                 output = torch.zeros([residual_total.size(0),self.pred_len,residual_total.size(2)],dtype=residual_total.dtype).to(residual_total.device)
#                 for i in range(self.channels):
#                     output[:,:,i] = self.Linear[i](residual_total[:,:,i])
#                 residual_total = output
#             else:
#                 residual_total = self.Linear(residual_total.permute(0,2,1)).permute(0,2,1)
                
                
#             output=residual_total+forecast_total.detach()
            
#         else:
#             output=forecast_total
            


#         # enc_out = self.enc_embedding(residual_total, x_mark_enc)

#         # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
#         # x_dec[:,:self.label_len,:]=residual_total[:,-self.label_len:,:]

#         # dec_out = self.dec_embedding(x_dec, x_mark_dec)
#         # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
#         # dec_out = self.projection(dec_out)
#         # output= dec_out[:,-self.pred_len:,:]+forecast_total
        
        
#         # dec_out = self.end_conv1(dec_out)
#         # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
#         if variation==True:

#             return output,forecastback_total,kl_divergence_total# [B, L, D],
            
#         else:

#             return output,forecastback_total# [B, L, D],
            
            
#     def get_item(self, x_enc, x_mark_enc, x_dec, x_mark_dec, nbeats,step=2,variation=False,
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        
        
#         if variation==True:
#             forecast_total=[]
#             forecastback_total=[]
#             residual_total=[]
            
            
#             kl_divergence_total=[]

#             for i in range(self.enc_in):            
#                 x_input=x_enc[:,:,i]
#                 forecast,forecastback,kl_divergence,store1,store2=nbeats.get_item(x_input,variation)
#                 forecast_total.append(forecast)
#                 forecastback_total.append(forecastback)
#                 # residual_total.append(residual)
#                 kl_divergence_total.append(kl_divergence)
     

#             forecast_total=torch.stack(forecast_total,axis=2)
#             forecastback_total=torch.stack(forecastback_total,axis=2)
            


            
#         else:
#             forecast_total=[]
#             forecastback_total=[]
#             residual_total=[]
            

#             for i in range(self.enc_in):            
#                 x_input=x_enc[:,:,i]
#                 forecast,forecastback=nbeats(x_input)
#                 forecast_total.append(forecast)
#                 forecastback_total.append(forecastback)
#                 # residual_total.append(residual)

#             forecast_total=torch.stack(forecast_total,axis=2)
#             forecastback_total=torch.stack(forecastback_total,axis=2)
            
#         # residual_total=torch.stack(residual_total,axis=2)
        
#         if step==2:
#             residual_total=x_enc-forecastback_total.detach()
#             enc_out = self.enc_embedding(residual_total.detach(), x_mark_enc)

#             enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
            
#             x_dec[:,:self.label_len,:]=residual_total[:,-self.label_len:,:]
#             x_dec=x_dec.detach()

#             dec_out = self.dec_embedding(x_dec, x_mark_dec)
#             dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
#             dec_out = self.projection(dec_out)
            
#             output= dec_out[:,-self.pred_len:,:]+forecast_total.detach()
            
#         else:
#             output=forecast_total
            
            
            

#         # enc_out = self.enc_embedding(residual_total, x_mark_enc)

#         # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
#         # x_dec[:,:self.label_len,:]=residual_total[:,-self.label_len:,:]

#         # dec_out = self.dec_embedding(x_dec, x_mark_dec)
#         # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
#         # dec_out = self.projection(dec_out)
#         # output= dec_out[:,-self.pred_len:,:]+forecast_total
        
        
#         # dec_out = self.end_conv1(dec_out)
#         # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
#         if variation==True:
#             if self.output_attention:
#                 return output, attns,forecastback_total,kl_divergence_total,store1,store2
            
#             else:
#                 return output,forecastback_total,kl_divergence_total,store1,store2       # [B, L, D],
            
#         else:
#             if self.output_attention:
#                 return output, attns,forecastback_total
            
#             else:
#                 return output,forecastback_total# [B, L, D],            
            
            
            
        
            
            

