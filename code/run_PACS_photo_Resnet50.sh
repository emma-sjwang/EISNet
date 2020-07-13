python train.py --batch_size 64 --n_classes 7 --network resnet50 --val_size 0.1 --folder_name test --nesterov False --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source art_painting cartoon photo --target sketch --image_size 222 -e 100 -g 0 -l 0.001 --nce_k 1024 --moco_weight 0.5 --k_triplet 256 --margin 2 --jig_weight 0.7 --alpha 0.999

