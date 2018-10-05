# Select qualified images from twitter dataset
import os
import re
import sys
from shutil import copyfile

dataset_dir = "../datasets/google/preprocessed"
images_per_class = 100

dir_cnt = 0
fail_cnt = 0
for subdir, dirs, files in os.walk(dataset_dir):
    if subdir == dataset_dir:
        continue
    nfiles = len([f for f in os.listdir(subdir) \
                  if os.path.isfile(os.path.join(subdir, f))])
    if nfiles < images_per_class:
        print("{} has {} images.".format(subdir, nfiles))
        fail_cnt += 1
    dir_cnt += 1

print("{} out of {} folders has less than {} images.".format(
         fail_cnt, dir_cnt, images_per_class))
