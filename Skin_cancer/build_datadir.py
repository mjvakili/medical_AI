'''
assuming that all the images in the HAM10000
data set are stored in the data directory, 
this code uses the HAM10000 meta data to transfer the 
images to a set of subdirectories, each corresponding 
to a cancer type
'''

from imutils import paths
import random
import shutil
import os
import pandas as pd
import numpy as np

data_dir = 'data/'

meta = pd.read_csv("HAM10000_metadata.csv")
cancer_type = meta.dx
cancer_id = meta.image_id

unique_types = np.unique(cancer_type) 

for utype in unique_types:
  
  os.rmdir(data_dir+utype) 
  os.mkdir(data_dir+utype)

image_files = list(paths.list_images('data/'))

for img in image_files:
  
  img_id = img.split('/')[1].split('.')[0]
  img_label = meta.dx[meta.image_id == img_id]
  img_label = img_label.values[0]
  destination = data_dir+img_label
  dest = shutil.move(str(img), destination) 

