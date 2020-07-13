# Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization
by [Shujun Wang](www.cse.cuhk.edu.hk/~sjwang), [Lequan Yu](https://yulequan.github.io/), Caizi Li, [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/), and [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/).

## Introduction
This repository is for our ECCV2020 paper 'Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization'.

## Requirements
-   python 3.6.1
-   torch 1.1.0
-   torch-geometric 1.2.1
-   Other packages in requirements.txt

## Usage
1. Clone the repository and set up the folders in the following structure:

  ├── code                   
 |   ├── CGC-Net
 |
 ├── data 
 |   ├── proto
 |        ├──mask (put the instance masks into this folder)    
 |             ├──"your-dataset"
 |                 ├──fold_1
 |                       ├──1_normal
 |                       ├──2_low_grade
 |                       ├──3_high_grade
 |                 ├──fold_2
 |                 ├──fold_3
 |
 |   ├── raw(put the images into this folder))	   
 |        ├──"your-dataset"
 |                 ├──fold_1
 |                       ├──1_normal
 |                       ├──2_low_grade
 |                       ├──3_high_grade
 |                 ├──fold_2
 |                 ├──fold_3
 ├── experiment	
 
2. 





▶ python train.py --batch_size 64 --n_classes 7 --network resnet18 --val_size 0.1 --folder_name test --nesterov False --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source art_painting cartoon sketch --target photo --image_size 222 -e 100 -g 2 -l 0.001 --nce_k 1024 --moco_weight 0.5 --k_triplet 256 --margin 2 --jig_weight 0.7 --alpha 0.999

▶ python train.py --batch_size 64 --n_classes 7 --network resnet50 --val_size 0.1 --folder_name test --nesterov False --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source art_painting cartoon photo --target sketch --image_size 222 -e 100 -g 3 -l 0.001 --nce_k 1024 --moco_weight 0.5 --k_triplet 256 --margin 2 --jig_weight 0.7 --alpha 0.999


