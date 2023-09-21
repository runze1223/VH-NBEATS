from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Basis_function,Basis_ETT_hour,Basis_ETT_minute
from torch.utils.data import DataLoader

import numpy as np 
import torch 
import os

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

data_basis = {
    'ETTh1': Basis_ETT_hour,
    'ETTh2': Basis_ETT_hour,
    'ETTm1': Basis_ETT_minute,
    'ETTm2': Basis_ETT_minute,
    'custom': Basis_function,
}



def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    return data_set, data_loader


#generate the hierarchical timestamp basis 
def basis_provider(args, flag):
    basis_loader = data_basis[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    freq = args.freq

    folder_path = './basis/' + args.data_path + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    basis = basis_loader(
    root_path=args.root_path,
    data_path=args.data_path,
    size=[args.seq_len, args.label_len, args.pred_len],
    features=args.features,
    target=args.target,
    timeenc=timeenc,
    freq=freq
        )
    basis_data=basis.generate()

    if args.features == 'M':
            if args.data=="ETTh2" or  args.data=="ETTh1":
                basis_data[3][:,:-1]=0

    basis_data[0]=np.array(basis_data[0])
    basis_data[1]=np.array(basis_data[1])
    basis_data[2]=np.array(basis_data[2])
    basis_data[3]=np.array(basis_data[3])

    np.save(folder_path + 'basis0.npy', basis_data[0])
    np.save(folder_path + 'basis1.npy', basis_data[1])
    np.save(folder_path + 'basis2.npy', basis_data[2])
    np.save(folder_path + 'basis3.npy', basis_data[3])

    return basis_data