# Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization
by [Shujun Wang](https://www.cse.cuhk.edu.hk/~sjwang), [Lequan Yu](https://yulequan.github.io/), Caizi Li, [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/), and [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/).

## Introduction
This repository is for our ECCV2020 paper '[Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization](https://link.springer.com/chapter/10.1007/978-3-030-58545-7_10)'.
![cellgraph](https://emma-sjwang.github.io/project_img/EISNet.png)
The framework of the proposed EISNet for domain generalization. We train a feature Encoder $f$ for discriminative and transferable feature extraction and a classifier for object recognition. Two complementary tasks, a momentum metric learning task and a self-supervised auxiliary task, are introduced to prompt general feature learning. We maintain a momentum updated Encoder (MuEncoder) to generate momentum updated embeddings stored in a large memory bank. Also, we design a $K$-hard negative selector to locate the informative hard triplets from the memory bank to calculate the triplet loss. The auxiliary self-supervised task predicts the order of patches within an image.

## Requirements
-   python 3.6.8 
   
   ```
   conda create -n EISNet python=3.6.8
   ```
   
-   PyTorch 1.4.0 
    
    ``` bash
    source activate EISNet 
    conda install pytorch==1.4.0 torchvision cudatoolkit=9.2 -c pytorch 
    ```
    
-   Other packages in requirements.txt


## Usage
1. Clone the repository and download the dataset [PACS](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk) and [VLCS](http://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file) into folder `EISNet/Dataset`.
> If you have already put the dataset into other paths, you need to change path in txt files in `EISNet/code/data/txt_lists/*.txt`.
2. Train the model.

    ``` bash
    sh run_PACS_photo_Resnet50.sh
    ```


## Citation
If EISNet is useful for your research, please consider citing:
```angular2html
@inproceedings{wang2020learning,
  title={Learning from Extrinsic and IntrinsicSupervisions for Domain Generalization},
  author={Wang, Shujun and Yu, Lequan and Li, Caizi and Fu, Chi-Wing and Heng, Pheng-Ann},
  booktitle={ECCV}, 
  year={2020}
}
```


