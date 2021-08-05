# Lightweight Image Super-Resolution by Multi-scale Aggregation

Pytorch implementation of "Lightweight Image Super-Resolution by Multi-scale Aggregation", IEEE Transactions on Broadcasting 2021


## Overview

![LMAN](/figs/arch.png)

The architecture of our proposed Lightweight Multi-scale Aggregation Network LMAN. The details about our proposed LMAN can be found in [our paper](https://ieeexplore.ieee.org/document/9233990).

Citation:

```latex
@ARTICLE{LMAN,
  author={Wan, Jin and Yin, Hui and Liu, Zhihao and Chong, Aixin and Liu, Yanting},
  journal={IEEE Transactions on Broadcasting}, 
  title={Lightweight Image Super-Resolution by Multi-Scale Aggregation}, 
  year={2021},
  volume={67},
  number={2},
  pages={372-382},
  doi={10.1109/TBC.2020.3028356}}
```

## Contents
1. [Train](#train)
2. [Test](#test)
3. [Results](#results)
4. [Acknowledgements](#acknowledgements)

## Train
### Begin to train

1. Cd to 'src/', run the following scripts to train models.
**You can use scripts in file 'demo' to train models for our paper.**

    ```bash
    # BI, scale 2, 3, 4, 8
    # LMAN Small-SCALE model (x2) 
    # python main.py --model LMAN --scale 2 --n_resgroups 4 --n_feats 64 --patch_size 96 --save LMAN_base_x2 --reset 

    # LMAN Small-SCALE model (x3) - from LMAN Small-SCALE model (x2)
    # python main.py --model LMAN --scale 3 --n_resgroups 4 --n_feats 64 --patch_size 144 --save LMAN_base_x3 --reset --pre_train ../experiment/LMAN_base_x2/model/model_best.pt

    # LMAN Small-SCALE model (x4) - from LMAN Small-SCALE model (x2)
    # python main.py --model LMAN --scale 4 --n_resgroups 4 --n_feats 64 --patch_size 192 --save LMAN_base_x4 --reset --pre_train ../experiment/LMAN_base_x2/model/model_best.pt


    # LMAN Small-SCALE model (x8) - from LMAN Small-SCALE model (x2)
    # python main.py --model LMAN --scale 8 --n_resgroups 4 --n_feats 64 --patch_size 384 --save LMAN_base_x8 --reset --pre_train ../experiment/LMAN_base_x2/model/model_best.pt

    # LMAN in the paper (x2)
    # python main.py --model LMAN --scale 2 --n_resgroups 16 --n_feats 64 --patch_size 96 --save LMAN_x2_16 --reset

    # LMAN in the paper (x3) - from LMAN (x2)
    # python main.py --model LMAN --scale 3 --n_resgroups 16 --n_feats 64 --patch_size 144 --save LMAN_x3_16 --reset --pre_train ../experiment/LMAN_x2_16/model/model_best.pt

    # LMAN in the paper (x4) - from LMAN (x2)
    # python main.py --model LMAN --scale 4 --n_resgroups 16 --n_feats 64 --patch_size 192 --save LMAN_x4_16 --reset --pre_train ../experiment/LMAN_x2_16/model/model_best.pt

    # LMAN in the paper (x8) - from LMAN (x2)
    # python main.py --model LMAN --scale 8 --n_resgroups 16 --n_feats 64 --patch_size 384 --save LMAN_x8_16 --reset --pre_train ../experiment/LMAN_x2_16/model/model_best.pt

    ```


## Test

1. Clone this repository:

   ```shell
   git clone https://github.com/jinwan1994/LMAN.git
   ```
2. All the models (BIX2/3/4/8) can be downloaded from [BaiduYun](https://pan.baidu.com/s/19ZbluRQVXKJl8umG9lv_-A)(bjtu), place the models to `./experiment/`. 

3. Cd to '/src', run the following scripts.

    **You can use scripts in file 'demo' to produce results for our paper.**

    ```bash
   # Standard benchmarks (Ex. LMAN_x2)
    # python main.py --model LMAN --save test_base_x2 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2 --n_resgroups 4 --n_feats 64 --pre_train ../experiment/model_base_x2.pt --test_only --save_results --save_gt  
    # python main.py --model LMAN --save test_x2 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2 --n_resgroups 16 --n_feats 64 --pre_train ../experiment/model_x2.pt --test_only --save_results --save_gt  

    # Standard benchmarks (Ex. LMAN_x3)
    # python main.py --model LMAN --save test_base_x3 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 3 --n_resgroups 4 --n_feats 64 --pre_train ../experiment/model_base_x3.pt --test_only #--save_results --save_gt 
    # python main.py --model LMAN --save test_x3 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 3 --n_resgroups 16 --n_feats 64 --pre_train ../experiment/model_x3.pt --test_only #--save_results --save_gt 

    # Standard benchmarks (Ex. LMAN_x4)
    # python main.py --model LMAN --save test_base_x4 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 4 --n_resgroups 4 --n_feats 64 --pre_train ../experiment/model_base_x4.pt --test_only # --save_results --save_gt
    # python main.py --model LMAN --save test_x4 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 4 --n_resgroups 16 --n_feats 64 --pre_train ../experiment/model_x4.pt --test_only # --save_results --save_gt

    # Standard benchmarks (Ex. LMAN_x8)
    # python main.py --model LMAN --save test_base_x8 --data_test Set5_x8+Set14_x8+B100_x8+Urban100_x8+Manga109_x8 --scale 8 --n_resgroups 4 --n_feats 64 --pre_train ../experiment/model_base_x8.pt  --test_only #--save_results --save_gt
    # python main.py --model LMAN --save test_x8 --data_test Set5_x8+Set14_x8+B100_x8+Urban100_x8+Manga109_x8 --scale 8 --n_resgroups 16 --n_feats 64 --pre_train ../experiment/model_x8.pt  --test_only #--save_results --save_gt

    ```
4. Finally, SR results and PSNR/SSIM values for test data are saved to `./experiment/*`. (PSNR/SSIM values in our paper are obtained using Matlab2015b)

## Results

#### Quantitative Results

![benchmark](/figs/result_1.png)

Benchmark SISR results. Average PSNR/SSIM for scale factor x2, x3, x4,and x8 on datasets Set5,Set14, Manga109, BSD100, and Urban100.

#### Visual Results

![visual](/figs/result_2.png)

Visual comparison for x8 SR on  Manga109 and Urban100. dataset.

## Acknowledgements

- This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes.

