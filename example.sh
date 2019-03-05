#!/bin/bash

# for training
python main.py --batch_size=10 --image_h=360 --image_w=480 --image_c=1 --image_dir=/root/datasets/artifact_images/databook.txt
#--image_dir=/root/yq/SegNet/CamVid/train.txt   --image_h=360 --image_w=480 --image_c=1
#--image_dir=/root/datasets/artifact_images/databook.txt 

# for finetune from saved ckpt
#python main.py --finetune=/root/yq/SegNet/logs/model.ckpt-0  --batch_size=5

#for testing
# python main.py --testing=/root/yq/SegNet/logs/model.ckpt-0 --batch_size=5 --save_image=True



##################################################################################################################################



#python main.py --log_dir=/tmp3/first350/TensorFlow/Logs/ --image_dir=/tmp3/first350/SegNet-Tutorial/CamVid/train.txt --val_dir=/tmp3/first350/SegNet-Tutorial/CamVid/val.txt --batch_size=5

# for finetune from saved ckpt
# python main.py --finetune=/tmp3/first350/TensorFlow/Logs/model.ckpt-1000  --log_dir=/tmp3/first350/TensorFlow/Logs/ --image_dir=/tmp3/first350/SegNet-Tutorial/CamVid/train.txt --val_dir=/tmp3/first350/SegNet-Tutorial/CamVid/val.txt --batch_size=5

#for testing
# python main.py --testing=/tmp3/first350/TensorFlow/Logs/model.ckpt-19000  --log_dir=/tmp3/first350/TensorFlow/Logs/ --test_dir=/tmp3/first350/SegNet-Tutorial/CamVid/test.txt --batch_size=5 --save_image=True
