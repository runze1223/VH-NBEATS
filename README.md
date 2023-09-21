# VH-NBEATS
### This is an offical implementation of VH-NBEATS and VH-PatchTST model. 



## Key Designs

:star2: 

:star2: 

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/model.png)

## Results

### Supervised Learning

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table3.png)

### Implement the project

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from . Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/vae```
```
sh ./scripts/vae/ETTH1.sh
```
You can adjust the hyperparameters based on your needs (e.g. different look-back windows and prediction lengths, beta,   ). We also provide codes for some baseline models inside the project.


## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer


## Contact



## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```

```

