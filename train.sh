python train.py \
    -net resnet50 \
    -image-size 224 \
    -pretrained \
    -batch 360 \
    -workers 32 \
    -lr 0.04 \
    -lr_step 20,40,60 \
    -epoch 80 \
    -warm 0 \
    -save_epoch 1 \
    -num_classes 2 \
    -gpus 0,1,2,3 \
    -cutmix_prob 0.5 \
    -apex \
    -imgs_root /home/data/terrorism/train_sqsh \
    -label_txt terror_sqsh_v2.2_train.txt \
    -resume \
    -refine checkpoints/resnet50_sqsh_v0/resnet50-80-regular.pth \
    # -opt_level O1 \
    # -test \
    # -test_label_txt scene_11cls_val.txt \
    # -imgs_root /home/jovyan/data/scene_11cls/train \
    # -label_txt scene_11cls_train.txt \
    # -test \
    # -test_label_txt scene_11cls_val.txt \
