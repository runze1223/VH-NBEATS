# VH-NBEATS
### This is an offical implementation of VH-NBEATS and VH-PatchTST model. 
## Key Designs

:star2: Hierarchical timestamp basis Block is developed to capture global perspective pattern of time series.

:star2: Variational autoencoder structure is introduced to achieve a significant improvement over the standard deterministic approach

![alt text](https://github.com/runze1223/VH-NBEATS/blob/main/pic/VH-NBEATS.png)

## Results


![alt text](https://github.com/runze1223/VH-NBEATS/blob/main/pic/Experiment_results.png)

## Visulization 

![alt text](https://github.com/runze1223/VH-NBEATS/blob/main/pic/Visulization.png)


### Implement the project

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download the ETTH1, ETTH2, Electricity and Traffic data from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and WTH data from [Google Drive](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR?usp=sharing) Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/vae```
```
sh ./scripts/vae/ETTH1.sh
```
You can adjust the hyperparameters based on your needs (e.g. different look-back windows and prediction lengths, beta and so on). We also provide codes for some baseline models inside the project.


## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/yuqinie98/PatchTST

https://github.com/cchallu/nbeatsx


## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```

```

