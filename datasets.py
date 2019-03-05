import random
import numpy as np
import cv2
#from SamplePreprocessor import preprocess
from glob import glob

import gzip
import pickle
import torch.utils.data as data
import os
#from utils import maybe_download
from os.path import join, basename, dirname, exists
from args import *
home = os.environ['HOME']

class ArtPrint(data.Dataset):
  '''artifact printings dataset'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images')

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/sh/tdd0784neuv9ysh/AABm3gxtjQIZ2R9WZ-XR9Kpra?dl=0',
    #               'iam_handwriting', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    #chars = set()
    self.samples = []
    ct=0
    for line in labelsFile:
      ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]
      # GT text are columns starting at 9
      labelPath = lineSplit[1]
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      if ct>=10000:
        break
  def __str__(self):
    return 'Artifact word image dataset. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    #gt=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label
    
    
if __name__=='__main__':
  artp=ArtPrint()
  leng=artp.__len__()
  print(leng)
  for idx in range(leng):
    img,label=artp.__getitem__(idx)
    if img.shape!=(32,128) or label.shape!=(32,128):
      print('-----')
      print(img.shape)
      print(label.shape)