# Deep-JSCC-for-images-with-OFDM

![Structure](Structure.png)    

## Datasets

This repository contains codes for CIFAR-10 and CelebA. For CelebA, you will need to download the dataset under `data` folder. You can also use other datasets but you need to customize the dataloader. One example is `data/CelebA_dataset.py`. 

## Train the model

All available options are under `options` folder. Change `--feedforward` for different models. For example, set feedforward as 'IMPLICIT' for IMPLICIT model in the paper. Set feedforward as 'EXPLICIT-RES' for EXPLICIT model in the paper. 

One example for training:

    python train.py --gpu_ids '0' --feedforward 'EXPLICIT-RES' --N_pilots 2 --n_downsample 2 --C_channel 12 --S 6 
      --SNR 20 --dataset_mode 'CIFAR10' --n_epochs 200 --n_epochs_decay 200 --lr 1e-3 
    
Suppose the input image has a size of C x W x H. To keep the size consistent, you would need to satisfy the requirement:  WH/(2^(2xn_downsample))xC_channel = Sx128

## Reference

> Mingyu Yang, Chenghong Bian, Hun-Seok Kim, "Deep Joint Source Channel Coding for WirelessImage Transmission with OFDM", accepted by ICC 2021

    @article{yang2021deep,
        title={Deep Joint Source Channel Coding for WirelessImage Transmission with OFDM},
        author={Yang, Mingyu and Bian, Chenghong and Kim, Hun-Seok},
        journal={arXiv preprint arXiv:2101.03909},
        year={2021}
    }
