from args import *

import tensorflow as tf
import os

from datasets import ArtPrint
from torch.utils.data import DataLoader, ConcatDataset, random_split#, SequentialSampler #yike: add SequentialSampler
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import model

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def checkArgs():
    if FLAGS.testing != '':
        print('The model is set to Testing')
        print("check point file: %s"%FLAGS.testing)
        print("CamVid testing dir: %s"%FLAGS.test_dir)
    elif FLAGS.finetune != '':
        print('The model is set to Finetune from ckpt')
        print("check point file: %s"%FLAGS.finetune)
        print("CamVid Image dir: %s"%FLAGS.image_dir)
        #print("CamVid Val dir: %s"%FLAGS.val_dir)
    else:
        print('The model is set to Training')
        print("Max training Iteration: %d"%FLAGS.max_steps)
        print("Initial lr: %f"%FLAGS.learning_rate)
        print("CamVid Image dir: %s"%FLAGS.image_dir)
        #print("CamVid Val dir: %s"%FLAGS.val_dir)

    print("Batch Size: %d"%FLAGS.batch_size)
    print("Log dir: %s"%FLAGS.log_dir)


def main(args):
    checkArgs()
    transform_train = transforms.Compose([

      transforms.Lambda(lambda img: cv2.resize(img, (FLAGS.image_w,FLAGS.image_h), interpolation=cv2.INTER_CUBIC)),
      transforms.Lambda(lambda img: np.expand_dims(img,3) ),
      #transforms.Lambda(lambda img: add_artifacts(img,args)),
      #transforms.Lambda(lambda img: cv2.transpose(img))
    ])
    arprint=ArtPrint(FLAGS.data_root, transform=transform_train)
    concat=arprint
    print(arprint)
    idxTrain = int( len(arprint)*0.9)
    trainset, testset = random_split(concat, [idxTrain, len(concat)-idxTrain])
    trainloader = DataLoader(trainset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True,num_workers=4)
#    validateloader=DataLoader(trainset, batch_size=args.batchsize, shuffle=False, drop_last=False,num_workers=2)
    testloader = DataLoader(testset, batch_size=FLAGS.batch_size, shuffle=False, drop_last=False,num_workers=2)
    print('kkk')
    print(len(trainloader))
    print(len(testloader))

#    for images,labels in trainloader:
#      img=labels.numpy()
#      print(img.shape)
#      print(type(images))
#      img=np.squeeze(img[3,:,:,0])
#      print(img.shape)
      
#      cv2.imwrite('/root/datasets/artifact_images/test2.jpg',img)
#      break
#    raise Exception('So far so good.')
    
    if FLAGS.testing:
        model.test(FLAGS)
    elif FLAGS.finetune:
        model.training(FLAGS, loader=trainloader,validateloader=testloader, is_finetune=True)
    else:
        model.training(FLAGS, loader=trainloader,validateloader=testloader, is_finetune=False)

if __name__ == '__main__':
  tf.app.run()
