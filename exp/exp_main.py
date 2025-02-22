from data_provider.data_factory import data_provider, basis_provider
from exp.exp_basic import Exp_Basic
from models import VNLinear, Informer, Autoformer, Transformer, DLinear, Linear, NLinear, nbeats,FEDformer,PatchTST,nbeats_embed,PatchTST_vae,nbeats_original
from utils.tools import EarlyStopping,EarlyStopping2, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import os
import time


import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)


    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'N-BEATS': nbeats_original,
            'V-NBEATS': nbeats,
            'VH-NBEATS': nbeats_embed,
            'VH-PatchTST': PatchTST_vae,
            'VNL': VNLinear,
            'PatchTST': PatchTST,
        }

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _get_basis(self, flag):
        basis_data= basis_provider(self.args, flag)
        return basis_data

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim


    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, train_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()


                if self.args.embed != 'timeF':
                    batch_x_mark = batch_x_mark.long().to(self.device)
                    batch_y_mark = batch_y_mark.long().to(self.device)
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif self.args.model=='Nbeats':
                            if self.args.variation== True:
                                outputs,backoutputs,kldivergence = self.model(batch_x)
                            else:
                                outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or self.args.model=="PatchTST":
                        outputs = self.model(batch_x)
                    elif self.args.model=='VH-NBEATS':
                        if self.args.variation== True:
                            outputs,kldivergence = self.model(batch_x,batch_x_mark, train_data)
                        else:
                            outputs = self.model(batch_x,batch_x_mark,train_data)
                    elif self.args.model=='VH-PatchTST' :
                        outputs,kldivergence = self.model(batch_x,batch_x_mark,train_data)
                            
                    elif self.args.model=='Nbeats' or self.args.model=='VNL':
                        if self.args.variation== True:
                            outputs,backoutputs,kldivergence = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting):
        basis_data=self._get_basis(flag='train')
        train_data, train_loader = self._get_data(flag='train')
        vali_data , vali_loader = self._get_data(flag='val')
        test_data , test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        # scheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=10, gamma=0.1)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
        #                             steps_per_epoch = train_steps,
        #                             pct_start = self.args.pct_start,
        #                             epochs = self.args.train_epochs,
        #                             max_lr = self.args.learning_rate)

        if self.args.is_training_second:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        else:
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    batch_y = batch_y.float().to(self.device)

                    if self.args.embed != 'timeF':
                        batch_x_mark = batch_x_mark.long().to(self.device)
                        batch_y_mark = batch_y_mark.long().to(self.device)
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)


                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            elif self.args.model=='Nbeats' or self.args.model=='VNL':
                                if self.args.variation== True:
                                    outputs,backoutputs,kldivergence = self.model(batch_x)
                                else:
                                    outputs = self.model(batch_x)

                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if 'Linear' in self.args.model or self.args.model=="PatchTST":
                            outputs = self.model(batch_x)

                        elif self.args.model=='VH-NBEATS' :
                            if self.args.variation== True:
                                outputs,kldivergence = self.model(batch_x,batch_x_mark,basis_data)
                            else:
                                outputs = self.model(batch_x,batch_x_mark,basis_data)  
                        elif self.args.model=='Nbeats' or self.args.model=='VNL':
                            if self.args.variation== True:
                                outputs,backoutputs,kldivergence = self.model(batch_x)
                            else:
                                outputs = self.model(batch_x)
                        elif self.args.model=='VH-PatchTST' :
                            outputs,kldivergence = self.model(batch_x,batch_x_mark,basis_data)

                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)


                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        if self.args.model=='Nbeats' or self.args.model=='VNL' or self.args.model=='VH-NBEATS' or self.args.model=='VH-PatchTST':

                            if self.args.variation==True:
                                loss1=criterion(outputs, batch_y)
                                loss =(loss1)+0.01*kldivergence
                            else:
                                loss=criterion(outputs, batch_y)
                                loss1=loss
                                loss2=loss
                                kldivergence=loss
                        else:
                            loss = criterion(outputs, batch_y)
                            loss1=loss
                            loss2=loss
                            kldivergence=loss

                        train_loss.append(loss1.item())

                    

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}, loss_back: {3:.7f},klD: {4:.7f}".format(i + 1, epoch + 1, loss1.item(),loss1.item(), loss1.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                        # scheduler.step()

                    # if self.args.lradj == 'TST':
                    #     adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    #     scheduler.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(basis_data, vali_loader, criterion)
                test_loss = self.vali(basis_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        basis_data=self._get_basis(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
  
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.embed != 'timeF':
                    batch_x_mark = batch_x_mark.long().to(self.device)
                    batch_y_mark = batch_y_mark.long().to(self.device)
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or self.args.model=="PatchTST":
                            outputs = self.model(batch_x)
                        elif self.args.model=='Nbeats' or self.args.model=='VNL':
                            if self.args.variation== True:
                                outputs,backoutputs,kldivergence = self.model(batch_x)
                            else:
                                outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or self.args.model=="PatchTST":
                            outputs = self.model(batch_x)
                    elif self.args.model=='Nbeats'  or self.args.model=='VNL':
                        if self.args.variation== True:
                            outputs,backoutputs,kldivergence = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)


                    elif self.args.model=='VH-NBEATS':
                            if self.args.variation== True:
                                outputs,kldivergence = self.model(batch_x,batch_x_mark,basis_data)
                            else:
                                outputs = self.model(batch_x,batch_x_mark,basis_data)  
                    elif self.args.model=='VH-PatchTST' :
                        outputs,kldivergence = self.model(batch_x,batch_x_mark,basis_data)

                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  
                true = batch_y  

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(os.path.join(folder_path, str(i) + '.pdf'),gt, pd)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()


        np.save(folder_path + 'pred.npy', preds[:100,:,:])
        np.save(folder_path + 'true.npy', trues[:100,:,:])
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')


        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        batch_y_marks=[]

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)


                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif self.args.model=='Nbeats' or self.args.model=='VNL':
                            if self.args.variation== True:
                                outputs,backoutputs,kldivergence = self.model(batch_x)
                            else:
                                outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    elif self.args.model=='Nbeats'  or self.args.model=='VNL':
                        if self.args.variation== True:
                            outputs,backoutputs,kldivergence = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                batch_y_mark=batch_y_mark.detach().cpu().numpy()
                batch_y_marks.append(batch_y_mark)


                pred = outputs.detach().cpu().numpy() -batch_y.detach().cpu().numpy() # .squeeze()
                preds.append(pred)




        batch_y_marks=np.array(batch_y_marks)
        batch_y_marks=batch_y_marks.reshape(-1, batch_y_marks.shape[-2], batch_y_marks.shape[-1])

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        np.save(folder_path + 'real_embed.npy', batch_y_marks)

        return
