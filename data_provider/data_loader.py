import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
#creating instance of one-hot-encoder
encoder = OneHotEncoder(handle_unknown='ignore')
warnings.filterwarnings('ignore')


#Generate the data and hierarchical timestamp index

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        #Generate the time index (Day:[0-23], Week:[0,163], Year:[0, 8783]) 

        if self.timeenc == 0:
            df_stamp['week'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1) 
            df_stamp['week']=24*df_stamp['week']+df_stamp['hour']  
            df_stamp['day_of_year'] = df_stamp.date.apply(lambda row: row.dayofyear, 1)

            df_stamp['day_of_year'] = 24*(df_stamp['day_of_year']-1)+df_stamp['hour']
            df_stamp['day_of_year']=df_stamp['day_of_year'].astype('category')
            df_stamp['week']=df_stamp['week'].astype('category')
            df_stamp['hour']=df_stamp['hour'].astype('category')
            data_stamp = df_stamp.drop(['date'], axis=1).values
            
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        data_stamp=data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)


        if self.timeenc == 0:
            df_stamp['week'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1) 
            df_stamp['week']=24*df_stamp['week']+df_stamp['hour']  
            df_stamp['day_of_year'] = df_stamp.date.apply(lambda row: row.dayofyear, 1)

            df_stamp['day_of_year'] = 24*(df_stamp['day_of_year']-1)+df_stamp['hour']
            df_stamp['day_of_year']=df_stamp['day_of_year'].astype('category')
            df_stamp['week']=df_stamp['week'].astype('category')
            df_stamp['hour']=df_stamp['hour'].astype('category')
            data_stamp = df_stamp.drop(['date'], axis=1).values
            
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()


        if self.data_path=="Weather_Gallipoli.csv":
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path),sep=";")
        else:

            df_raw = pd.read_csv(os.path.join(self.root_path,
                                self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        

        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)

            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)


        if self.timeenc == 0:
            df_stamp['week'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1) 
            df_stamp['week']=24*df_stamp['week']+df_stamp['hour']  
            df_stamp['day_of_year'] = df_stamp.date.apply(lambda row: row.dayofyear, 1)

            df_stamp['day_of_year'] = 24*(df_stamp['day_of_year']-1)+df_stamp['hour']
            df_stamp['day_of_year']=df_stamp['day_of_year'].astype('category')
            df_stamp['week']=df_stamp['week'].astype('category')
            df_stamp['hour']=df_stamp['hour'].astype('category')
            data_stamp = df_stamp.drop(['date'], axis=1).values
            
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

#moving average kernel
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w        


#generate the gradient of the basis function 
def generate_the_gradient(data_x_train,kernel_size=None):

    if kernel_size==None:
        data_difference=np.diff(data_x_train,axis=0)
    else:
        a,b=data_x_train.shape
        front = data_x_train[ 0:1, :].repeat((kernel_size-1)//2+(kernel_size-1)%2,axis=0)
        end = data_x_train[ -1:, :].repeat( (kernel_size-1)//2, axis=0)
        data_x_train = np.concatenate([front, data_x_train, end], axis=0)
        data_convolved =np.concatenate( [moving_average(data_x_train[:,i],kernel_size).reshape(-1,1) for i in range(b)],axis=1)
        data_difference =np.diff(data_convolved,axis=0)

    return data_difference



#Generate the hierarchical timestamp basis function
class Basis_function(Dataset):
    def __init__(self, root_path, size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        type_map = {'train': 0, 'val': 1, 'test': 2}
        flag='train'
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        if self.data_path=="Weather_Gallipoli.csv":
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path),sep=";")
        else:

            df_raw = pd.read_csv(os.path.join(self.root_path,
                                self.data_path))
            
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)

            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_x_train=data[border1:border2]
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_raw['week'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_raw['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_raw['new']=24*df_raw['week']+df_raw['hour']
        df_raw['new']=df_raw['new'].astype('category')
        df_raw['month'] = df_stamp.date.apply(lambda row: row.dayofyear, 1)

        df_raw['year']=24*(df_raw['month']-1)+df_raw['hour']
        df_raw['year']=df_raw['year'].astype('category')
        df_raw['week']=df_raw['week'].astype('category')
        df_raw['hour']=df_raw['hour'].astype('category')
        df_raw['month']=df_raw['month'].astype('category')
        
        
        df_raw_train=df_raw[border1:border2]

        data_difference_24=generate_the_gradient(data_x_train,24)
        data_difference=generate_the_gradient(data_x_train)
        data_difference_168=generate_the_gradient(data_x_train,168)


        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-5]=data_difference
        store=df_raw_None.groupby("new").mean(numeric_only=True)
        basis=store.cumsum(axis = 0)
        store=np.array(store)
        basis=np.array(basis)
        last=-basis[-1,:]/basis.shape[0]
        last=np.expand_dims(last,axis=0)
        store= store+last
        basis=store.cumsum(axis = 0)
        basis=np.array(basis)
        self.week_day = basis
        self.week_day_g = store

        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-5]=data_difference_24
        store=df_raw_None.groupby("new").mean(numeric_only=True)
        basis=store.cumsum(axis = 0)
        store=np.array(store)
        basis=np.array(basis)
        last=-basis[-1,:]/basis.shape[0]
        last=np.expand_dims(last,axis=0)
        store= store+last
        basis=store.cumsum(axis = 0)
        basis=np.array(basis)
        self.week = basis
        self.week_g = store

        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-5]=data_difference
        store=df_raw_None.groupby("hour").mean(numeric_only=True)
        basis=store.cumsum(axis = 0)
        store=np.array(store)
        basis=np.array(basis)
        last=-basis[-1,:]/basis.shape[0]
        last=np.expand_dims(last,axis=0)
        store= store+last
        basis=store.cumsum(axis = 0)
        basis=np.array(basis)
        self.day = basis
        self.day_g = store

        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-5]=data_difference_168
        store=df_raw_None.groupby("year").mean(numeric_only=True)
        basis=store.cumsum(axis = 0)
        store=np.array(store)
        basis=np.array(basis)
        new=basis[-25,:]
        new_store=np.zeros_like(new)
        new=np.tile(new,(24,1))
        basis[-24:,:]=new
        new_store=np.tile(new_store,(24,1))
        store[-24:,:]=new_store
        last=-basis[-1,:]/basis.shape[0]
        last=np.expand_dims(last,axis=0)
        store= store+last
        basis=store.cumsum(axis = 0)
        self.year = basis
        self.year_g = store

    def generate(self):
        basis_list=[]
        basis_list.append(self.week_day)
        basis_list.append(self.week)
        basis_list.append(self.day)
        basis_list.append(self.year)

        return basis_list


class Basis_ETT_hour(Dataset):
    def __init__(self, root_path, size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        type_map = {'train': 0, 'val': 1, 'test': 2}
        flag='train'
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
                

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        border1s = [0, 12 * 30 * 24 - self.seq_len+145, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24+145, 12 * 30 * 24 + 4 * 30 * 24+145, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        df_stamp = df_raw[['date']]


        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_raw = df_raw[['date',self.target]]

            
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_x_train=data[border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_raw['week'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_raw['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_raw['new']=24*df_raw['week']+df_raw['hour']
        df_raw['new']=df_raw['new'].astype('category')
        df_raw['month'] = df_stamp.date.apply(lambda row: row.dayofyear, 1)

        df_raw['year']=24*(df_raw['month']-1)+df_raw['hour']
        df_raw['year']=df_raw['year'].astype('category')
        df_raw['week']=df_raw['week'].astype('category')
        df_raw['hour']=df_raw['hour'].astype('category')
        df_raw['month']=df_raw['month'].astype('category')
               
        df_raw_train=df_raw[border1:border2]

        data_difference_24=generate_the_gradient(data_x_train,24)
        data_difference=generate_the_gradient(data_x_train)
        data_difference_168=generate_the_gradient(data_x_train,168)
        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-5]=data_difference
        store=df_raw_None.groupby("new").mean(numeric_only=True)
        basis=store.cumsum(axis = 0)

        store=np.array(store)
        basis=np.array(basis)
        last=-basis[-1,:]/basis.shape[0]
        last=np.expand_dims(last,axis=0)
        store= store+last
        basis=store.cumsum(axis = 0)
        basis=np.array(basis)
        self.week_day = basis

        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-5]=data_difference_24
        store=df_raw_None.groupby("new").mean(numeric_only=True)
        basis=store.cumsum(axis = 0)
        store=np.array(store)
        basis=np.array(basis)
        last=-basis[-1,:]/basis.shape[0]
        last=np.expand_dims(last,axis=0)
        store= store+last
        basis=store.cumsum(axis = 0)
        basis=np.array(basis)
        self.week = basis

        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-5]=data_difference
        store=df_raw_None.groupby("hour").mean(numeric_only=True)
        basis=store.cumsum(axis = 0)
        store=np.array(store)
        basis=np.array(basis)
        last=-basis[-1,:]/basis.shape[0]
        last=np.expand_dims(last,axis=0)
        store= store+last
        basis=store.cumsum(axis = 0)
        basis=np.array(basis)
        self.day = basis

        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-5]=data_difference_168
        store=df_raw_None.groupby("year").mean(numeric_only=True)

        basis=store.cumsum(axis = 0)
        store=np.array(store)
        basis=np.array(basis)
        last=-basis[-1,:]/basis.shape[0]
        last=np.expand_dims(last,axis=0)
        store= store+last
        basis=store.cumsum(axis = 0)
        basis=np.array(basis)
        
        self.year = basis

    def generate(self):
        basis_list=[]
        basis_list.append(self.week_day)
        basis_list.append(self.week)
        basis_list.append(self.day)
        basis_list.append(self.year)
        return basis_list

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[2]
        border2 = border2s[2]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
       
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.timeenc=0
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

        self.data_stamp = data_stamp[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        seq_y_mark = self.data_stamp[r_begin:r_begin + self.label_len]

    
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Basis_ETT_minute(Dataset):
    def __init__(self, root_path, size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        type_map = {'train': 0, 'val': 1, 'test': 2}
        flag='train'
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
                

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)

            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_x_train=data[border1:border2]
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_raw['week'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_raw['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_raw['minutes'] = df_stamp.date.apply(lambda row: row.minute, 1)
        df_raw['minutes']=df_raw['minutes']/15
        

        df_raw['new']=24*4*df_raw['week']+df_raw['hour']*4+df_raw['minutes']
        df_raw['new']=df_raw['new'].astype('category')
        df_raw['month'] = df_stamp.date.apply(lambda row: row.dayofyear, 1)

        df_raw['year']=24*(df_raw['month']-1)*4+df_raw['hour']*4+df_raw['minutes']

        df_raw['hour']=df_raw['hour']*4+df_raw['minutes']
        df_raw['year']=df_raw['year'].astype('category')
        df_raw['week']=df_raw['week'].astype('category')
        df_raw['hour']=df_raw['hour'].astype('category')
        df_raw['month']=df_raw['month'].astype('category')
        df_raw['minutes']=df_raw['minutes'].astype('category')

        df_raw_train=df_raw[border1:border2]

        data_difference_24=generate_the_gradient(data_x_train,24*4)
        data_difference=generate_the_gradient(data_x_train)
        data_difference_168=generate_the_gradient(data_x_train,24*4*7)


        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-6]=data_difference
        
        store=df_raw_None.groupby("new").mean(numeric_only=True)
        basis=store.cumsum(axis = 0)
        basis=np.array(basis)
        self.week_day = basis

        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-6]=data_difference_24
        store=df_raw_None.groupby("new").mean(numeric_only=True)
        basis=store.cumsum(axis = 0)
        basis=np.array(basis)
        self.week = basis

        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-6]=data_difference
        store=df_raw_None.groupby("hour").mean(numeric_only=True)
        basis=store.cumsum(axis = 0)
        basis=np.array(basis)
        self.day = basis

        df_raw_None = df_raw_train.iloc[1:,:].copy()
        df_raw_None.iloc[:,1:-6]=data_difference_168
        store=df_raw_None.groupby("year").mean(numeric_only=True)

        basis=store.cumsum(axis = 0)

        basis=np.array(basis)
        self.year = basis

    def generate(self):
        basis_list=[]
        basis_list.append(self.week_day)
        basis_list.append(self.week)
        basis_list.append(self.day)
        basis_list.append(self.year)
        return basis_list
