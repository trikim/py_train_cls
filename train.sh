# cd /project/train/src_repo
# python /project/train/src_repo/train.py \
python train.py \
    -net mobilenet \
    -batch 64 \
    -image-size 128 \
    -lr 0.04 \
    -num_classes 7 \
    -gpus 0 \
    -lr_step 20,40,60 \
    -epoch 60 \
    -cutmix_prob 0.5 \
    -imgs_root /home/data/14 \
    -label_txt clothes_train.txt \
    -pretrained \
    -test \
    -opt_level O0 \
    -export_onnx \
    # -resume \
    # -refine mobilenet-80-regular.pth
# wait
# python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --scale_values=[58.395,57.120,57.375] --mean_values=[123.675,116.28,103.53] --input_model /project/train/models/mobilenet/deploy.onnx --data_type=FP16 --output_dir /project/train/models/mobilenet/
