# Deep-JSCC-for-images-with-OFDM

![Structure](example2.png)    

## Datasets

This repository contains codes for CIFAR-10 and CelebA. For CelebA, you will need to download the dataset under `data` folder. You can also use other datasets but you need to customize the dataloader. One example is `data/CelebA_dataset.py`. 

## Train the model

All available options are under `options` folder. Change `--feedforward` for different models. For example, set feedforward as 'IMPLICIT' for IMPLICIT model in the paper. Set feedforward as 'EXPLICIT-RES' for EXPLICIT model in the paper. 

One example for training:

